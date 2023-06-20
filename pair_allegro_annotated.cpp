/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include <pair_allegro.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

// TODO: Only if MPI is available
#include <mpi.h>


//////////////////////////////////////////////////////////////////////////////////////////////
// KS: Can we do a code patch to the original CUDA-RASPA code just like how Allegro did here?
//     Check LAMMPS instruction for writing new pair styles
//     https://docs.lammps.org/Developer_write_pair.html
//////////////////////////////////////////////////////////////////////////////////////////////


// We have to do a backward compatability hack for <1.10
// https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4
// Basically, the check in torch::jit::freeze
// (see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// is wrong, and we have ro "reimplement" the function
// to get around that...
// it's broken in 1.8 and 1.9
// BUT the internal logic in the function is wrong in 1.10
// So we only use torch::jit::freeze in >=1.11
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
  #define DO_TORCH_FREEZE_HACK
  // For the hack, need more headers:
  #include <torch/csrc/jit/passes/freeze_module.h>
  #include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
  #include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
  #include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#endif


using namespace LAMMPS_NS;

// KS: Set up some flags for LAMMPS constructor
//    Here not writing binary restart files; many-body interactions type
PairAllegro::PairAllegro(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  if(const char* env_p = std::getenv("ALLEGRO_DEBUG")){
    std::cout << "PairAllegro is in DEBUG mode, since ALLEGRO_DEBUG is in env\n";
    debug_mode = 1;
  }

// KS: Check device
  if(torch::cuda::is_available()){
    int deviceidx = -1;
    if(comm->nprocs > 1){
      MPI_Comm shmcomm;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
          MPI_INFO_NULL, &shmcomm);
      int shmrank;
      MPI_Comm_rank(shmcomm, &shmrank);
      deviceidx = shmrank;
    }
    if(deviceidx >= 0) {
      int devicecount = torch::cuda::device_count();
      if(deviceidx >= devicecount) {
        if(debug_mode) {
          // To allow testing multi-rank calls, we need to support multiple ranks with one GPU
          std::cerr << "WARNING (Allegro): my rank (" << deviceidx << ") is bigger than the number of visible devices (" << devicecount << "), wrapping around to use device " << deviceidx % devicecount << " again!!!";
          deviceidx = deviceidx % devicecount;
        }
        else {
          // Otherwise, more ranks than GPUs is an error
          std::cerr << "ERROR (Allegro): my rank (" << deviceidx << ") is bigger than the number of visible devices (" << devicecount << ")!!!";
          error->all(FLERR,"pair_allegro: mismatch between number of ranks and number of available GPUs");
        }
      }
    }
    device = c10::Device(torch::kCUDA,deviceidx);
  }
  else {
    device = torch::kCPU;
  }
  std::cout << "Allegro is using device " << device << "\n";
}


// KS: Required destructor to delete all memory that was allocated by the pair style 
PairAllegro::~PairAllegro(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

// KS: Requirement of LAMMPS for requesting neighbor list through init_style
void PairAllegro::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Allegro requires atom IDs");

  // need a full neighbor list
  // KS: “full” neighbor list is both atoms of a pair are listed in each other’s neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  // KS: requesting ghost atoms.
  //    Ghost atoms are typically employed in parallel algorithms such as the spatial decomposition technique, 
  //    where the simulation domain is divided into smaller subdomains or cells that are assigned to different processors. 
  //    During the simulation, each processor possesses a local copy of the atoms within its assigned subdomain and 
  //    a set of ghost atoms, which are replicas of neighboring atoms from adjacent subdomains.
  neighbor->requests[irequest]->ghost = 1;

  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Allegro requires newton pair on");
}


// KS: This is where potential parameters are checked for completeness, derived parameters computed
//    Another purpose of the init_one() function is to symmetrize the potential parameter arrays.
//    i, j represent pair i and j
double PairAllegro::init_one(int i, int j)
{ 
  // KS: return cutoff for any i,j pair which means cutoff is the same for all i,j pair
  return cutoff;
}


// KS: allocate new potential parameters
void PairAllegro::allocate()
{
  allocated = 1;
  int n = atom->ntypes;  // KS: ntypes is atom types 

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
}


// KS: The arguments to the settings() function are the arguments given to the pair_style command in the LAMMPS input file
void PairAllegro::settings(int narg, char ** /*arg*/) {
  // "allegro" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command, too many arguments");
}


// KS: The arguments to the coeff() function are the arguments to the pair_coeff command in the LAMMPS input file
void PairAllegro::coeff(int narg, char **arg) {
  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  // KS: <type1> <type2> here are species names of atom types specified in Allegro training YAML
  //      In order, the names of the Allegro model's atom types to use for LAMMPS atom types 1, 2, and so on. 
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients, should be * * <model>.pth <type1> <type2> ... <typen>");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  std::vector<std::string> elements(ntypes);
  for(int i = 0; i < ntypes; i++){
    elements[i] = arg[i+1];
  }

  std::cout << "Allegro: Loading model from " << arg[2] << "\n";

  std::unordered_map<std::string, std::string> metadata = {
    {"config", ""},
    {"nequip_version", ""},
    {"r_max", ""},
    {"n_species", ""},
    {"type_names", ""},
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""},
    {"allow_tf32", ""}
  };

  // KS: Load deployed Allegro model from TorchScript file
  //     metadata is a dictionary. it will load data from deployed model
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  // KS: Setting to eval() mode is necessary to rescale the output (energy, forces) back to its original scale
  model.eval();

  // Check if model is a NequIP model
  // KS: metadata is loaded previously from the deployed Allegro model
  if (metadata["nequip_version"].empty()) {
    error->all(FLERR, "The indicated TorchScript file does not appear to be a deployed NequIP model; did you forget to run `nequip-deploy`?");
  }

  // KS: Just keep this part of code in CUDA-RASPA
  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (model.hasattr("training")) {
    std::cout << "Allegro: Freezing TorchScript model...\n";
    #ifdef DO_TORCH_FREEZE_HACK
      // Do the hack
      // Copied from the implementation of torch::jit::freeze,
      // except without the broken check
      // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
      bool optimize_numerics = true;  // the default
      // the {} is preserved_attrs
      auto out_mod = freeze_module(
        model, {}
      );
      // See 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
      auto graph = out_mod.get_method("forward").graph();
      OptimizeFrozenGraph(graph, optimize_numerics);
      model = out_mod;
    #else
      // Do it normally
      model = torch::jit::freeze(model);
    #endif
  }

  // KS: Some JIT setting seems to avoid recompilation of the model due to the variation in the input shapes
  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    // Set JIT bailout to avoid long recompilations for many steps
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty()) {
      // This is the default used in the Python code
      jit_bailout_depth = 2;
    } else {
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
    }
    torch::jit::getBailoutDepth() = jit_bailout_depth;
  #else
    // In PyTorch >=1.11, this is now set_fusion_strategy
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty()) {
      // This is the default used in the Python code
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
    } else {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while(std::getline(strat_stream, fusion_type, ',')) {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

  // Set whether to allow TF32:
  // KS: A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs.
  bool allow_tf32;
  if (metadata["allow_tf32"].empty()) {
    // Better safe than sorry
    allow_tf32 = false;
  } else {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  // std::cout << "Allegro: Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }


  // KS: Cutoff radius for interatomic interactions used in Allegro training
  //    As defined in init_one(), cutoff is same for all ij pairs
  cutoff = std::stod(metadata["r_max"]);

  //TODO: This
  // KS: map Allegro type (species name) to LAMMPS type (0,1,2..)
  type_mapper.resize(ntypes);
  std::stringstream ss;
  int n_species = std::stod(metadata["n_species"]);
  ss << metadata["type_names"];    // KS: store type_names in Allegro to ss
  std::cout << "Type mapping:" << "\n";
  std::cout << "Allegro type | Allegro name | LAMMPS type | LAMMPS name" << "\n";
  for (int i = 0; i < n_species; i++){
    std::string ele;
    ss >> ele;

    for (int itype = 1; itype <= ntypes; itype++){

      // KS: if ele from Allegro is equal to LAMMPS pair_style setting (value = 0)
      //    Note - Species set in Allegro YAML configs should correspond to the order set in LAMMPS pair_style
      if (ele.compare(arg[itype + 3 - 1]) == 0){
        type_mapper[itype-1] = i;
        std::cout << i << " | " << ele << " | " << itype << " | " << arg[itype + 3 - 1] << "\n";
      }
    }
  }

  // set setflag i,j for type pairs where both are mapped to elements
  // KS: setflag==1 means ij pair interactions coeff have been set.
  //     Here it means type mapping is success
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
        if ((type_mapper[i] >= 0) && (type_mapper[j] >= 0))
            setflag[i][j] = 1;

  // KS: retrieve environement variable "BATCH SIZE"
  char *batchstr = std::getenv("BATCHSIZE");
  if (batchstr != NULL) {
    batch_size = std::atoi(batchstr);
  }

}

// Force and energy computation
// KS: workhorse of pair style. This is where we have the nested loops over all pairs 
//    of particles from the neighbor list to compute forces and - if needed - energies and virials.
void PairAllegro::compute(int eflag, int vflag){

  // KS: ev_init() initializes several flags derived from the eflag and vflag parameters 
  //    signaling whether the energy and virial need to be tallied and whether only globally or also per-atom.
  ev_init(eflag, vflag);

  // Get info from lammps:
  // KS: If I understand correctly, all properties here are stored in terms of 
  //    a sub-domain in the simulation box, as this is how LAMMPS partition the system 
  //    for parallization.

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // KS: Number of local/real atoms in a sub-domain
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  // Should be on.
  int newton_pair = force->newton_pair;

  // KS: Number of neighbor lists in the sub-domain 
  int inum = list->inum;
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // KS: Number of ghost atoms neighbors for the sub-domain
  int nghost = list->gnum;
  // Total number of atoms in the current sub-domain
  int ntotal = inum + nghost;

  // Mapping from neigh list ordering to x/f ordering
  // KS: array of (local) indices of atoms for which neighbor lists have been created
  int *ilist = list->ilist;
  // KS: an inum sized array; each element in this array represents the number of entries in the corresponding neighbor list
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  // KS: a list of pointers to those neighbor lists.
  int **firstneigh = list->firstneigh;


  // Total number of bonds (sum of number of neighbors)
  int nedges = 0;

  // Number of bonds per atom
  // KS: initialize a vector; it contains true number of neigh atoms (<cutoff) for atom i
  //    see below
  std::vector<int> neigh_per_atom(nlocal, 0);

#pragma omp parallel for reduction(+:nedges)
  // KS: loop over all local/real atoms in the sub-domain
  for(int ii = 0; ii < nlocal; ii++){
    int i = ilist[ii];

    int jnum = numneigh[i];                           // KS: total number of neighboring atoms of atom i
    int *jlist = firstneigh[i];                       // KS: neighbor list array for atom i

    // KS: loop over all neighboring atoms of atom i
    for(int jj = 0; jj < jnum; jj++){                 
      int j = jlist[jj];                              // KS: mapping neighbor list index to local atom index
      j &= NEIGHMASK;                                 // KS: step to avoid segmentation fault, problem with LAMMPS itself

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if(rsq <= cutoff*cutoff) {
        neigh_per_atom[ii]++;                         // KS: accumulate TRUE number of neigh atoms for atom i (r < cutoff)
        nedges++;                                     // KS: accumulate number of edges (ij pairs) in the subdomain
      }
    }
  }






  // Cumulative sum of neighbors, for knowing where to fill in the edges tensor
  // KS: variables are initialized with value 0 by default?
  std::vector<int> cumsum_neigh_per_atom(nlocal);

  for(int ii = 1; ii < nlocal; ii++){
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii-1] + neigh_per_atom[ii-1];     // KS: accumulate neighbors 
  }


  // KS: initialize feature vectors
  // KS: Atomic position tensor, first dimension is the total number of atoms (real + ghost), second dimension is x, y, z coords
  torch::Tensor pos_tensor = torch::zeros({ntotal, 3});
  // KS: Edge tensor, nedges are the total number of ij pair bonds in the subdomain, type is Int64
   // This is basically the neighbor list. In allegro Python, they used `ase.neighborlist.primitive_neighbor_list` function 
   // to generate neighbor list, here, the implementation should give the same results of ase neighbor list function.
   // The first row dimension i is sorted in ascending order, but second dimension j is not sorted.
  torch::Tensor edges_tensor = torch::zeros({2,nedges}, torch::TensorOptions().dtype(torch::kInt64));
  // KS: 
  torch::Tensor ij2type_tensor = torch::zeros({ntotal}, torch::TensorOptions().dtype(torch::kInt64));


  // KS: 
  auto pos = pos_tensor.accessor<float, 2>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto ij2type = ij2type_tensor.accessor<long, 1>();


  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  if (debug_mode) printf("Allegro edges: i j rij\n");
#pragma omp parallel for
  for(int ii = 0; ii < ntotal; ii++){
    int i = ilist[ii];          // KS: index conversion
    int itag = tag[i];          // KS: extract atom id
    int itype = type[i];        // KS: atom type

    ij2type[i] = type_mapper[itype - 1];     // KS: central atom type (convert lammps atom type to that of Allegro)

    pos[i][0] = x[i][0];              // KS: assign positions to pos_tensor
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];

    if(ii >= nlocal){continue;}


    // KS: handling real atom
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];

    int edge_counter = cumsum_neigh_per_atom[ii];
    for(int jj = 0; jj < jnum; jj++){
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if(rsq > cutoff*cutoff) {continue;}

      // TODO: double check order
      edges[0][edge_counter] = i;
      edges[1][edge_counter] = j;

      edge_counter++;

      if (debug_mode) printf("%d %d %.10g\n", itag-1, jtag-1, sqrt(rsq));
    }
  }
  if (debug_mode) printf("end Allegro edges\n");

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("atom_types", ij2type_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);

  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<float, 2>();

  //torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu(); WRONG WITH MPI

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
  float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];

  //std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  //std::cout << "Total energy: " << total_energy_tensor << "\n";
  //std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  //std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

// KS: In MOF system, we either take total_energy_tensor or sum over all atomic energies. Because Eij != Eji, so atomic energy of framework atoms that are 
  // close to adsorbate molecule may be non-zero. This means we cannot simply sum over adsorbate atoms.  

  // Write forces and per-atom energies (0-based tags here)
  eng_vdwl = 0.0;
#pragma omp parallel for reduction(+:eng_vdwl)
  for(int ii = 0; ii < ntotal; ii++){
    int i = ilist[ii];

    f[i][0] = forces[i][0];
    f[i][1] = forces[i][1];
    f[i][2] = forces[i][2];
    if (eflag_atom && ii < inum) eatom[i] = atomic_energies[i][0];
    if(ii < inum) eng_vdwl += atomic_energies[i][0];
  }
}
