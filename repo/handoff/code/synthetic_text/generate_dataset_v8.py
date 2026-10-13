"""
generate_dataset_v8.py — Large-K Synthetic Text for Warm-Start Multiplication

KEY DESIGN: K=200 categories, T=10000 → T/K=50 (starved regime)

Why K=200 works:
  - With 200 categories, LinUCB needs to learn ~200 arm directions
  - Each arm gets ~50 pulls on average under uniform exploration
  - With d=200 features and low coherence (random-projected BoW → α̂ ≈ 0.07)
    each pull only informs about THAT arm → cold LinUCB is SLOW
  - LLM (Flan-T5/Qwen) classifies from raw text → concentrates on
    reward-relevant arms → provides directional warm-start

Categories: 200 categories from 10 broad domains (20 per domain)
  - Broad enough that LLM can attempt classification from text
  - Specific enough that keywords are distinctive
  - Some cross-domain overlap creates confusion (heavy gap tail)

Dataset structure (same as v3/v6/v7):
  - doc_features: (N, d) projected unit-norm features  
  - arm_features: (K, d) projected unit-norm arm features
  - texts: raw document strings for LLM classification
  - ys: (N,) true category index
  - category_names: list of K category strings
"""

import numpy as np
import random
import gzip
import pickle
from numpy.linalg import norm

# ================================================================
# 200 CATEGORIES: 10 domains × 20 categories each
# Each category has 15 keywords (distinctive phrases)
# ================================================================

CATEGORIES = {
    # DOMAIN 1: PHYSICS (20)
    "Quantum Mechanics": [
        "wave function collapse", "Heisenberg uncertainty principle",
        "quantum entanglement correlation", "superposition state measurement",
        "Schrödinger equation evolution", "photon polarization angle",
        "tunneling barrier probability", "Planck constant quantization",
        "spin angular momentum", "quantum decoherence environment",
        "Hilbert space dimension", "probability amplitude interference",
        "Bohr model transition", "Bell inequality violation", "EPR paradox nonlocality"
    ],
    "Classical Mechanics": [
        "Newton laws of motion", "Lagrangian mechanics variational",
        "Hamiltonian phase space", "rigid body rotation inertia",
        "orbital mechanics Kepler", "oscillation damping resonance",
        "conservation momentum collision", "central force problem",
        "gravitational potential energy", "projectile motion trajectory",
        "Coriolis fictitious force", "torque angular acceleration",
        "mechanical energy conservation", "Euler rigid body equations", "d'Alembert principle"
    ],
    "Electromagnetism": [
        "Maxwell equations electromagnetic", "Coulomb electric field charge",
        "Faraday electromagnetic induction", "magnetic dipole moment",
        "Ampere circuital law current", "Gauss divergence electric flux",
        "electromagnetic wave propagation", "Lorentz force moving charge",
        "capacitance dielectric material", "inductance solenoid coil",
        "Poynting vector energy flow", "waveguide resonant cavity",
        "antenna radiation pattern", "polarization transverse wave", "skin depth conductor"
    ],
    "Thermodynamics": [
        "entropy irreversible process", "Carnot cycle maximum efficiency",
        "Gibbs free energy spontaneous", "enthalpy heat content",
        "Boltzmann statistical distribution", "partition function canonical",
        "phase transition critical point", "heat capacity specific",
        "adiabatic expansion cooling", "isothermal compression work",
        "chemical potential equilibrium", "Clausius inequality statement",
        "van der Waals real gas", "Joule-Thomson throttling effect",
        "Maxwell thermodynamic relations"
    ],
    "Optics": [
        "refraction Snell law interface", "diffraction single slit pattern",
        "interference constructive destructive", "laser stimulated emission coherent",
        "fiber optic total internal reflection", "holography recording reconstruction",
        "nonlinear optics frequency doubling", "polarization Brewster angle",
        "Fresnel reflection coefficients", "Gaussian beam propagation waist",
        "optical cavity resonator modes", "aberration chromatic spherical",
        "Fourier transform spectroscopy", "Raman scattering molecular vibration",
        "photonic crystal bandgap"
    ],
    "Nuclear Physics": [
        "nuclear fission chain reaction", "fusion plasma confinement tokamak",
        "radioactive decay half-life", "alpha particle emission helium",
        "beta decay neutrino electron", "gamma ray electromagnetic nuclear",
        "nuclear binding energy mass defect", "shell model magic numbers",
        "neutron cross section capture", "nuclear reactor moderator coolant",
        "isotope separation enrichment", "nuclear waste disposal storage",
        "cosmic ray spallation", "nuclear magnetic resonance spin",
        "Mössbauer effect recoilless"
    ],
    "Particle Physics": [
        "Higgs boson mass generation", "quark confinement strong force",
        "lepton family electron muon", "Standard Model gauge symmetry",
        "neutrino oscillation mass hierarchy", "antimatter CP violation",
        "hadron collider proton beam", "Feynman diagram vertex calculation",
        "spontaneous symmetry breaking vacuum", "dark matter WIMP candidate",
        "supersymmetry sparticle prediction", "grand unification theory GUT",
        "string theory extra dimensions", "baryon asymmetry leptogenesis",
        "quantum chromodynamics gluon"
    ],
    "Condensed Matter": [
        "superconductor Cooper pair BCS", "semiconductor band gap doping",
        "topological insulator surface state", "Bose-Einstein condensate ultracold",
        "ferromagnet domain wall hysteresis", "antiferromagnet Néel temperature",
        "phonon lattice vibration crystal", "exciton binding energy optical",
        "Kondo effect magnetic impurity", "quantum Hall effect Landau level",
        "spin-orbit coupling Rashba", "Mott insulator correlation",
        "Weyl semimetal node", "skyrmion magnetic texture", "Anderson localization disorder"
    ],
    "Fluid Dynamics": [
        "turbulence Reynolds number transition", "Navier-Stokes viscous equation",
        "boundary layer separation detachment", "vortex shedding Kármán street",
        "Bernoulli pressure streamline", "compressible shock wave Mach",
        "laminar Poiseuille pipe flow", "drag coefficient aerodynamic",
        "cavitation bubble collapse", "surface tension capillary meniscus",
        "Rayleigh-Taylor instability mixing", "magnetohydrodynamic Alfvén wave",
        "microfluidic lab chip channel", "non-Newtonian shear thinning",
        "Stokes creeping flow sphere"
    ],
    "Astrophysics": [
        "black hole event horizon singularity", "neutron star pulsar magnetar",
        "dark energy accelerating expansion", "cosmic microwave background anisotropy",
        "gravitational wave LIGO detection", "galaxy rotation curve dark matter",
        "supernova explosion nucleosynthesis", "quasar active galactic nucleus",
        "Hubble constant expansion rate", "Big Bang nucleosynthesis primordial",
        "stellar evolution main sequence", "accretion disk X-ray binary",
        "gamma ray burst relativistic jet", "exoplanet transit detection method",
        "gravitational lensing magnification"
    ],
    "Geophysics": [
        "seismic wave P-wave S-wave", "mantle convection plate tectonics",
        "geomagnetic field reversal polarity", "Earth core iron nickel",
        "isostasy crustal buoyancy equilibrium", "gravity anomaly Bouguer",
        "heat flow geothermal gradient", "paleomagnetism polar wander",
        "earthquake focal mechanism tensor", "lithosphere asthenosphere boundary",
        "GPS geodetic deformation measurement", "magnetotelluric conductivity imaging",
        "seismic tomography velocity structure", "glacial isostatic adjustment rebound",
        "tidal forcing deformation solid"
    ],
    "Acoustics": [
        "sound wave longitudinal compression", "resonance standing wave frequency",
        "Doppler effect source motion", "acoustic impedance matching",
        "reverberation room decay time", "noise cancellation destructive interference",
        "ultrasound imaging transducer frequency", "sonar underwater detection ranging",
        "musical instrument harmonic overtone", "acoustic phonetics vocal formant",
        "sound absorption coefficient material", "diffraction sound barrier edge",
        "acoustic metamaterial negative index", "infrasound atmospheric monitoring",
        "psychoacoustics loudness perception"
    ],

    # DOMAIN 2: CHEMISTRY (20)
    "Organic Chemistry": [
        "carbon-carbon bond formation coupling", "nucleophilic substitution SN2 mechanism",
        "electrophilic aromatic substitution", "Grignard reagent organometallic",
        "Diels-Alder cycloaddition diene", "cross-coupling palladium catalyst Suzuki",
        "functional group transformation protection", "stereochemistry chiral enantiomer",
        "retrosynthetic analysis disconnection", "radical chain reaction initiation",
        "elimination E2 mechanism base", "oxidation reduction alcohol aldehyde",
        "amide bond peptide coupling", "click chemistry triazole formation",
        "asymmetric catalysis enantioselective"
    ],
    "Inorganic Chemistry": [
        "coordination complex ligand field", "transition metal oxidation state",
        "crystal field splitting energy", "organometallic sandwich compound",
        "bioinorganic metalloenzyme active site", "solid state ionic conductor",
        "cluster compound metal-metal bond", "main group element hydride",
        "lanthanide contraction periodic trend", "actinide nuclear fuel chemistry",
        "Werner coordination theory isomer", "chelate effect stability constant",
        "hard soft acid base HSAB", "electron transfer inner outer sphere",
        "magnetic susceptibility paramagnetism"
    ],
    "Physical Chemistry": [
        "reaction rate constant Arrhenius", "chemical equilibrium Le Chatelier",
        "molecular orbital theory LCAO", "spectroscopy absorption emission",
        "surface adsorption Langmuir BET", "electrochemistry Nernst potential",
        "quantum chemistry ab initio", "statistical thermodynamics ensemble",
        "photochemistry excited state quenching", "colloid surface chemistry DLVO",
        "kinetic molecular theory gas", "chemical bonding valence band",
        "intermolecular force van der Waals", "phase diagram binary eutectic",
        "transport phenomenon diffusion viscosity"
    ],
    "Analytical Chemistry": [
        "chromatography separation column HPLC", "mass spectrometry ionization fragmentation",
        "NMR spectroscopy chemical shift coupling", "infrared spectroscopy vibrational mode",
        "UV-visible absorption Beer-Lambert", "X-ray fluorescence elemental analysis",
        "electrochemical sensor voltammetry", "atomic absorption spectroscopy flame",
        "titration endpoint equivalence point", "gravimetric precipitation analysis",
        "capillary electrophoresis migration", "ion chromatography conductivity detection",
        "Raman spectroscopy fingerprint region", "thermal analysis DSC TGA",
        "sample preparation extraction cleanup"
    ],
    "Biochemistry": [
        "enzyme kinetics Michaelis-Menten", "protein folding secondary structure",
        "DNA replication polymerase mechanism", "RNA transcription promoter regulation",
        "lipid membrane bilayer fluidity", "carbohydrate glycosylation pathway",
        "signal transduction cascade phosphorylation", "metabolic pathway citric acid cycle",
        "cofactor coenzyme NAD FAD", "allosteric regulation conformational",
        "antibody antigen binding affinity", "gene expression regulation epigenetic",
        "protein-protein interaction complex", "amino acid sequence primary structure",
        "enzyme inhibitor competitive uncompetitive"
    ],
    "Polymer Chemistry": [
        "polymerization chain addition condensation", "molecular weight distribution PDI",
        "glass transition temperature Tg", "crystallinity degree melting point",
        "copolymer block random alternating", "crosslinking network vulcanization",
        "polymer degradation thermal oxidative", "elastomer rubber mechanical property",
        "conducting polymer conjugated backbone", "dendrimer branched architecture generation",
        "polymer brush surface grafting", "self-assembly block copolymer morphology",
        "biodegradable polymer PLA PGA", "polyelectrolyte charged solution",
        "polymer rheology viscoelastic modulus"
    ],
    "Electrochemistry": [
        "battery lithium ion intercalation", "fuel cell membrane electrode assembly",
        "corrosion passivation oxide layer", "electrolysis water splitting hydrogen",
        "supercapacitor double layer charge", "cyclic voltammetry redox peak",
        "Nernst equation cell potential", "overpotential activation concentration",
        "solid electrolyte interface SEI", "photoelectrochemistry semiconductor junction",
        "electrodeposition metal plating", "impedance spectroscopy equivalent circuit",
        "electrocatalysis oxygen reduction", "flow battery vanadium redox",
        "electrochemical sensor biosensor glucose"
    ],
    "Environmental Chemistry": [
        "ozone depletion CFC stratosphere", "greenhouse gas carbon dioxide methane",
        "acid rain sulfur dioxide nitrogen", "water treatment disinfection chlorination",
        "soil contamination remediation heavy metal", "air pollution particulate matter PM2.5",
        "pesticide residue bioaccumulation food chain", "plastic pollution microplastic ocean",
        "radioactive contamination nuclear fallout", "eutrophication algal bloom nutrient",
        "persistent organic pollutant PCB dioxin", "wastewater biological treatment activated sludge",
        "groundwater contamination plume transport", "photochemical smog ozone VOC NOx",
        "carbon capture sequestration storage"
    ],

    # DOMAIN 3: BIOLOGY (20)
    "Cell Biology": [
        "mitosis cell division chromosome", "endoplasmic reticulum protein folding",
        "mitochondria ATP oxidative phosphorylation", "cytoskeleton actin microtubule",
        "cell membrane receptor signaling", "apoptosis programmed cell death",
        "endocytosis vesicle membrane trafficking", "nucleus chromatin gene expression",
        "Golgi apparatus secretory pathway", "lysosome degradation autophagy",
        "stem cell differentiation pluripotent", "cell cycle checkpoint regulation",
        "tight junction cell adhesion", "ribosome translation mRNA protein",
        "peroxisome oxidation detoxification"
    ],
    "Genetics": [
        "DNA mutation point frameshift", "Mendelian inheritance dominant recessive",
        "gene regulation promoter enhancer", "CRISPR genome editing Cas9",
        "epigenetic methylation histone modification", "chromosome translocation deletion",
        "polymerase chain reaction PCR amplification", "genetic linkage recombination mapping",
        "population genetics allele frequency drift", "quantitative trait locus QTL",
        "transgenic organism gene knockout", "gene therapy viral vector delivery",
        "single nucleotide polymorphism SNP", "genomic imprinting parent-of-origin",
        "telomere shortening aging senescence"
    ],
    "Ecology": [
        "ecosystem food web trophic level", "biodiversity species richness evenness",
        "succession primary secondary climax", "population dynamics carrying capacity",
        "predator-prey Lotka-Volterra cycle", "mutualism symbiosis obligate",
        "competition exclusion niche overlap", "nutrient cycling nitrogen phosphorus",
        "keystone species ecosystem engineer", "island biogeography species-area",
        "metapopulation connectivity patch", "disturbance regime fire flood",
        "invasive species displacement native", "habitat fragmentation corridor",
        "climate change species range shift"
    ],
    "Microbiology": [
        "bacteria cell wall peptidoglycan", "virus capsid genome replication",
        "antibiotic resistance mechanism plasmid", "biofilm formation quorum sensing",
        "pathogen virulence factor invasion", "immune evasion antigenic variation",
        "microbiome gut community diversity", "fermentation anaerobic metabolism",
        "bacteriophage lytic lysogenic cycle", "horizontal gene transfer conjugation",
        "extremophile thermophilic halophilic", "nitrogen fixation rhizobium symbiosis",
        "prion misfolded protein disease", "mycology fungal hyphae spore",
        "protozoan parasite life cycle"
    ],
    "Neuroscience": [
        "neuron action potential synapse", "neurotransmitter dopamine serotonin",
        "brain plasticity learning memory", "cortical area functional localization",
        "hippocampus memory consolidation spatial", "prefrontal cortex executive function",
        "basal ganglia motor control habit", "cerebellum coordination timing",
        "amygdala fear emotion conditioning", "neural circuit connectivity mapping",
        "EEG brainwave oscillation rhythm", "fMRI BOLD signal activation",
        "axon myelination conduction velocity", "glial cell astrocyte oligodendrocyte",
        "neurodegeneration Alzheimer tau amyloid"
    ],
    "Immunology": [
        "antibody immunoglobulin B-cell humoral", "T-cell receptor antigen MHC presentation",
        "innate immunity macrophage neutrophil", "adaptive immunity clonal selection",
        "cytokine inflammatory response interleukin", "complement cascade membrane attack",
        "vaccine adjuvant immune memory", "autoimmune disease self-tolerance breakdown",
        "allergy hypersensitivity IgE mast cell", "transplant rejection graft host",
        "immune checkpoint PD-1 cancer therapy", "dendritic cell antigen processing",
        "mucosal immunity IgA secretory", "lymph node germinal center affinity",
        "immunodeficiency HIV AIDS virus"
    ],
    "Evolutionary Biology": [
        "natural selection fitness adaptation", "speciation reproductive isolation barrier",
        "phylogenetics molecular clock divergence", "genetic drift bottleneck founder",
        "sexual selection mate choice ornament", "convergent evolution analogous structure",
        "coevolution arms race mutualism", "fossil record transitional form",
        "homology common ancestor morphology", "adaptive radiation ecological niche",
        "gene flow migration population structure", "molecular evolution neutral theory",
        "evo-devo Hox gene body plan", "horizontal gene transfer prokaryote",
        "extinction mass background rate"
    ],
    "Plant Biology": [
        "photosynthesis chloroplast light reaction", "stomata gas exchange transpiration",
        "auxin growth hormone tropism", "root uptake nutrient water transport",
        "xylem phloem vascular tissue", "flower pollination seed fruit development",
        "germination dormancy breaking light", "circadian clock photoperiodism flowering",
        "defense mechanism secondary metabolite", "mycorrhizal symbiosis nutrient exchange",
        "nitrogen fixation legume nodule", "cell wall cellulose lignin",
        "meristem growth differentiation pattern", "drought stress ABA stomatal closure",
        "C4 CAM photosynthesis adaptation"
    ],
    "Marine Biology": [
        "coral reef bleaching thermal stress", "deep sea hydrothermal vent community",
        "plankton phytoplankton zooplankton bloom", "whale migration communication song",
        "shark predator apex trophic", "jellyfish bloom population explosion",
        "mangrove coastal ecosystem nursery", "seagrass meadow carbon sequestration",
        "ocean acidification calcification impact", "bioluminescence deep water communication",
        "fish school behavior collective movement", "intertidal zone exposure adaptation",
        "marine protected area conservation", "algae kelp forest productivity",
        "turtle nesting migration navigation"
    ],

    # DOMAIN 4: EARTH SCIENCE (20)
    "Volcanology": [
        "magma chamber eruption trigger", "pyroclastic flow ignimbrite deposit",
        "lava flow viscosity composition", "volcanic plume ash cloud aviation",
        "caldera collapse resurgence cycle", "lahar mudflow debris volcanic",
        "fumarole hydrothermal alteration gas", "shield volcano basalt Hawaii",
        "stratovolcano explosive andesitic", "cinder cone scoria tephra",
        "volcanic tremor seismic monitoring", "phreatic explosion steam driven",
        "volcanic island arc subduction", "hotspot mantle plume track",
        "supervolcano Yellowstone eruption risk"
    ],
    "Oceanography": [
        "thermohaline circulation deep water", "ocean current Gulf Stream eddy",
        "upwelling nutrient coastal productivity", "El Niño Southern Oscillation",
        "sea surface temperature anomaly satellite", "tsunami generation propagation coastal",
        "tidal cycle gravitational lunar solar", "ocean mixed layer thermocline depth",
        "salinity halocline freshwater input", "internal wave density stratification",
        "mesoscale eddy vortex transport", "continental shelf slope sediment",
        "sea ice Arctic Antarctic extent", "ocean heat content warming",
        "marine sediment core paleoclimate proxy"
    ],
    "Meteorology": [
        "weather forecast numerical model prediction", "thunderstorm convection lightning",
        "tornado supercell mesocyclone funnel", "hurricane typhoon tropical cyclone",
        "jet stream upper level trough ridge", "frontal system warm cold occluded",
        "precipitation rain snow hail formation", "atmospheric pressure isobar gradient",
        "fog visibility dew point condensation", "wind speed Beaufort scale gust",
        "climate zone Köppen classification", "monsoon seasonal rain reversal",
        "drought dry spell Palmer index", "heat wave urban heat island",
        "radar Doppler wind velocity reflectivity"
    ],
    "Glaciology": [
        "glacier flow ice sheet dynamics", "ice core paleoclimate temperature record",
        "sea level rise ice mass loss", "firn compaction densification snow",
        "calving iceberg terminus retreat", "moraine till glacial deposit landform",
        "permafrost thaw carbon release methane", "crevasse formation stress strain",
        "glacial erosion U-valley cirque", "ice shelf buttressing collapse",
        "albedo feedback ice-albedo warming", "subglacial lake water drainage",
        "Heinrich event ice rafted debris", "glacial-interglacial Milankovitch cycle",
        "polar ice cap Greenland Antarctic"
    ],
    "Paleontology": [
        "fossil preservation mineralization cast", "dinosaur extinction Cretaceous asteroid",
        "Cambrian explosion diversification fauna", "trilobite arthropod Paleozoic",
        "ammonite cephalopod biostratigraphy zone", "mammal evolution Cenozoic radiation",
        "hominin fossil bipedal brain size", "plant fossil Carboniferous coal",
        "mass extinction Permian recovery", "trace fossil ichnology behavior",
        "amber preservation insect Cretaceous", "micropaleontology foraminifera foram",
        "evolutionary transition fish tetrapod", "biogeography continental distribution",
        "taphonomy decay burial preservation"
    ],
    "Hydrology": [
        "watershed river basin drainage area", "groundwater aquifer recharge discharge",
        "flood frequency return period analysis", "evapotranspiration water balance",
        "precipitation runoff infiltration", "stream hydrograph peak discharge",
        "water table level fluctuation seasonal", "karst limestone cave dissolution",
        "sediment transport erosion deposition", "dam reservoir storage capacity",
        "irrigation water use allocation", "drought severity duration frequency",
        "wetland riparian buffer ecosystem", "snow melt spring runoff timing",
        "urban hydrology impervious surface"
    ],
    "Mineralogy": [
        "crystal structure unit cell symmetry", "mineral hardness Mohs scale",
        "silicate framework chain sheet", "feldspar quartz mica composition",
        "ore mineral deposit formation", "metamorphic grade pressure temperature",
        "igneous crystallization Bowen series", "clay mineral smectite kaolinite",
        "gemstone color clarity cut carat", "X-ray diffraction mineral identification",
        "optical mineralogy polarized light", "isomorphous substitution solid solution",
        "mineral cleavage fracture habit", "hydrothermal alteration vein deposit",
        "biomineralization shell tooth bone"
    ],
    "Soil Science": [
        "soil horizon A B C profile", "organic matter humus decomposition",
        "clay particle cation exchange capacity", "soil pH acidic alkaline nutrient",
        "compaction bulk density porosity", "erosion water wind tillage",
        "nitrogen phosphorus potassium fertility", "microbial activity respiration carbon",
        "soil moisture field capacity wilting", "pedogenesis soil formation factors",
        "texture sand silt clay triangle", "aggregate stability structure",
        "salinization sodic reclamation drainage", "carbon sequestration soil organic",
        "contamination heavy metal remediation phyto"
    ],

    # DOMAIN 5: COMPUTER SCIENCE (20)
    "Machine Learning": [
        "neural network gradient descent backpropagation", "supervised learning classification regression",
        "reinforcement learning reward policy value", "generalization overfitting regularization",
        "feature engineering selection extraction", "ensemble bagging boosting random forest",
        "kernel method support vector machine", "Bayesian inference prior posterior",
        "dimensionality reduction PCA embedding", "clustering K-means hierarchical density",
        "cross-validation hyperparameter tuning", "bias-variance tradeoff model complexity",
        "transfer learning domain adaptation", "attention mechanism transformer self",
        "loss function optimization convergence"
    ],
    "Computer Vision": [
        "image classification convolutional neural network", "object detection bounding box YOLO",
        "semantic segmentation pixel label", "face recognition embedding verification",
        "optical flow motion estimation video", "image generation GAN diffusion",
        "pose estimation skeleton keypoint", "depth estimation stereo monocular",
        "image super-resolution upscaling", "visual question answering multimodal",
        "scene understanding panoptic segmentation", "edge detection contour Canny",
        "feature matching SIFT descriptor", "video understanding temporal action",
        "3D reconstruction point cloud mesh"
    ],
    "Natural Language Processing": [
        "language model perplexity generation", "sentiment analysis opinion mining",
        "named entity recognition tagging", "machine translation sequence-to-sequence",
        "text summarization extractive abstractive", "question answering reading comprehension",
        "word embedding Word2Vec GloVe contextual", "dependency parsing syntax tree",
        "coreference resolution mention linking", "topic modeling LDA latent",
        "text classification document categorization", "relation extraction knowledge graph",
        "dialogue system chatbot response", "speech recognition ASR acoustic",
        "information retrieval ranking relevance"
    ],
    "Distributed Systems": [
        "consensus Paxos Raft leader election", "fault tolerance replication availability",
        "CAP theorem consistency partition network", "distributed database sharding partition",
        "message queue publish subscribe asynchronous", "microservice architecture service mesh",
        "load balancing horizontal scaling", "distributed lock mutual exclusion",
        "eventual consistency conflict resolution CRDT", "MapReduce parallel processing batch",
        "stream processing real-time event", "service discovery registration health check",
        "distributed tracing observability span", "circuit breaker retry timeout",
        "vector clock causality ordering"
    ],
    "Cryptography": [
        "encryption symmetric AES block cipher", "public key RSA elliptic curve",
        "hash function SHA collision resistance", "digital signature authentication integrity",
        "key exchange Diffie-Hellman protocol", "zero-knowledge proof verification",
        "homomorphic encryption computation ciphertext", "quantum cryptography QKD BB84",
        "blockchain consensus proof-of-work", "secure multiparty computation secret sharing",
        "certificate authority PKI trust chain", "random number generator entropy",
        "post-quantum lattice-based code-based", "side-channel timing power analysis",
        "oblivious transfer protocol"
    ],
    "Operating Systems": [
        "process scheduling preemptive priority", "virtual memory paging segmentation",
        "file system inode directory metadata", "device driver kernel module",
        "thread synchronization mutex semaphore", "deadlock prevention detection avoidance",
        "interrupt handler context switch", "memory management allocation fragmentation",
        "system call interface user kernel", "disk scheduling seek rotation latency",
        "container virtualization namespace cgroup", "bootloader initialization firmware BIOS",
        "security access control permission", "I/O multiplexing epoll select",
        "real-time scheduling deadline guarantee"
    ],
    "Database Systems": [
        "SQL query optimization execution plan", "transaction ACID isolation concurrency",
        "index B-tree hash structure performance", "normalization schema design redundancy",
        "NoSQL document key-value graph", "replication master-slave high availability",
        "query processing join algorithm hash", "storage engine LSM-tree B-tree",
        "distributed transaction two-phase commit", "data warehouse OLAP cube aggregation",
        "column store compression analytics", "time-series database IoT sensor",
        "graph database traversal property model", "cache invalidation consistency TTL",
        "backup recovery point-in-time restore"
    ],
    "Computer Networks": [
        "TCP congestion control window protocol", "routing BGP OSPF path selection",
        "DNS resolution domain name server", "HTTP REST API request response",
        "firewall packet filtering rule", "VPN tunnel encryption remote",
        "load balancer reverse proxy distribution", "SDN software-defined control plane",
        "WiFi wireless 802.11 access point", "5G cellular network millimeter wave",
        "CDN content delivery edge cache", "NAT port translation address",
        "HTTPS TLS certificate handshake", "WebSocket real-time bidirectional",
        "QoS quality service bandwidth latency"
    ],
    "Software Engineering": [
        "agile scrum sprint retrospective", "continuous integration deployment pipeline",
        "version control git branch merge", "code review pull request quality",
        "testing unit integration end-to-end", "design pattern singleton factory observer",
        "refactoring clean code technical debt", "microservice API gateway service",
        "requirements specification user story", "architecture monolith modular layered",
        "DevOps infrastructure automation IaC", "documentation API reference specification",
        "performance profiling optimization bottleneck", "monitoring alerting SLO incident",
        "dependency management package versioning"
    ],

    # DOMAIN 6: MATHEMATICS (20)
    "Linear Algebra": [
        "matrix eigenvalue eigenvector decomposition", "vector space basis dimension span",
        "linear transformation kernel image", "determinant inverse singular",
        "SVD singular value decomposition", "orthogonal projection Gram-Schmidt",
        "trace rank nullity theorem", "positive definite quadratic form",
        "Jordan canonical form nilpotent", "tensor product multilinear",
        "Hermitian unitary matrix spectral", "least squares pseudoinverse",
        "matrix exponential differential equation", "sparse matrix iterative solver",
        "Kronecker product block structure"
    ],
    "Probability Theory": [
        "random variable distribution density", "expectation variance moment generating",
        "conditional probability Bayes theorem", "central limit theorem convergence",
        "Markov chain transition stationary", "martingale stopping time optional",
        "large deviation concentration inequality", "stochastic process Brownian motion",
        "independence joint marginal copula", "law of large numbers convergence",
        "characteristic function Fourier transform", "measure theory sigma algebra",
        "ergodic theorem stationary mixing", "Poisson process arrival rate",
        "random walk recurrence transience"
    ],
    "Differential Equations": [
        "ordinary differential equation initial value", "partial differential equation boundary",
        "Laplace equation harmonic function", "heat equation diffusion parabolic",
        "wave equation propagation hyperbolic", "existence uniqueness Lipschitz Picard",
        "linear system matrix exponential", "stability Lyapunov equilibrium",
        "bifurcation parameter qualitative change", "numerical method Runge-Kutta Euler",
        "separation variables Fourier series", "Green function fundamental solution",
        "nonlinear dynamics chaos attractor", "variational formulation weak solution",
        "Sturm-Liouville eigenvalue problem"
    ],
    "Topology": [
        "open set continuous mapping homeomorphism", "compact space Heine-Borel cover",
        "connected component path simply", "fundamental group homotopy loop",
        "manifold atlas chart transition", "homology cohomology exact sequence",
        "covering space deck transformation", "fiber bundle section connection",
        "fixed point Brouwer Lefschetz", "knot invariant Jones polynomial",
        "simplicial complex triangulation CW", "dimension topological Hausdorff",
        "separation axiom Hausdorff normal", "quotient space identification relation",
        "Euler characteristic genus surface"
    ],
    "Number Theory": [
        "prime number distribution theorem", "modular arithmetic congruence residue",
        "Diophantine equation integer solution", "quadratic reciprocity Legendre symbol",
        "elliptic curve rational point", "Riemann zeta function zeros hypothesis",
        "algebraic number field extension", "p-adic absolute value completion",
        "class field theory abelian extension", "Fermat last theorem Wiles proof",
        "Goldbach conjecture twin prime", "continued fraction approximation",
        "lattice point geometry Minkowski", "cryptographic integer factorization",
        "analytic continuation L-function"
    ],
    "Combinatorics": [
        "graph coloring chromatic number", "counting permutation combination",
        "generating function power series", "Ramsey theory partition regularity",
        "extremal graph Turán density", "matroid independence greedy",
        "probabilistic method Lovász local lemma", "Catalan number binary tree",
        "partition integer Young tableau", "design combinatorial block BIBD",
        "matching bipartite Hall theorem", "flow network max-flow min-cut",
        "random graph Erdős-Rényi threshold", "coding theory error-correcting Hamming",
        "Polya enumeration burnside group action"
    ],
    "Optimization": [
        "convex optimization gradient descent convergence", "linear programming simplex interior point",
        "integer programming branch bound cut", "stochastic gradient momentum Adam",
        "constraint optimization Lagrange multiplier KKT", "dynamic programming Bellman equation",
        "metaheuristic simulated annealing genetic", "multi-objective Pareto front trade-off",
        "semidefinite programming relaxation SDP", "combinatorial optimization NP-hard approximation",
        "submodular function greedy guarantee", "online optimization regret bound",
        "proximal operator splitting ADMM", "nonconvex landscape saddle point",
        "distributed optimization consensus gradient"
    ],
    "Numerical Analysis": [
        "finite element method mesh assembly", "interpolation polynomial spline error",
        "quadrature numerical integration Gauss", "root finding Newton bisection",
        "eigenvalue algorithm QR Lanczos", "iterative solver conjugate gradient GMRES",
        "condition number stability ill-posed", "floating point arithmetic precision error",
        "Monte Carlo sampling variance reduction", "fast Fourier transform butterfly",
        "sparse linear system preconditioner", "adaptive refinement error estimator",
        "finite difference scheme stability CFL", "multigrid V-cycle coarse correction",
        "tensor decomposition Tucker CP"
    ],

    # DOMAIN 7: ENGINEERING (16)
    "Control Systems": [
        "feedback controller PID proportional integral", "transfer function Laplace domain pole",
        "state space representation observable controllable", "stability Routh-Hurwitz Nyquist",
        "root locus gain variation", "Bode plot frequency response margin",
        "optimal control LQR Riccati", "robust control H-infinity uncertainty",
        "nonlinear control Lyapunov sliding mode", "adaptive control parameter estimation",
        "model predictive control MPC horizon", "digital control Z-transform sampling",
        "sensor fusion Kalman filter estimation", "actuator saturation anti-windup",
        "system identification frequency step"
    ],
    "Signal Processing": [
        "Fourier transform frequency spectrum analysis", "filter design FIR IIR low-pass",
        "sampling theorem Nyquist aliasing", "wavelet transform time-frequency resolution",
        "noise reduction SNR Wiener filter", "adaptive filter LMS RLS convergence",
        "spectral estimation periodogram Welch", "compressed sensing sparse recovery",
        "beamforming array antenna direction", "modulation demodulation AM FM QAM",
        "audio signal speech enhancement", "image filtering convolution edge",
        "radar signal pulse compression detection", "biomedical signal ECG EEG EMG",
        "digital signal processor DSP real-time"
    ],
    "Robotics": [
        "robot kinematics forward inverse joint", "path planning RRT A-star obstacle",
        "SLAM simultaneous localization mapping", "manipulation grasping force control",
        "mobile robot navigation odometry", "computer vision depth perception",
        "reinforcement learning robot policy", "human-robot interaction safety collaboration",
        "swarm robotics collective behavior", "soft robot compliant actuator",
        "drone UAV autonomous flight control", "legged locomotion balance walking",
        "sensor fusion lidar camera IMU", "robot operating system ROS middleware",
        "industrial automation pick place assembly"
    ],
    "Materials Science": [
        "steel alloy heat treatment tempering", "composite fiber matrix laminate",
        "ceramic oxide sintering fracture", "polymer blend morphology compatibility",
        "thin film deposition CVD PVD", "nanomaterial quantum dot nanotube",
        "fatigue crack propagation S-N curve", "corrosion galvanic pitting crevice",
        "biomaterial implant biocompatibility tissue", "shape memory alloy transformation",
        "semiconductor silicon wafer fabrication", "superconductor critical temperature current",
        "magnetic material hysteresis coercivity", "thermal barrier coating turbine",
        "additive manufacturing 3D printing layer"
    ],
    "Electrical Engineering": [
        "circuit analysis Kirchhoff voltage current", "transistor MOSFET amplifier switching",
        "power electronics converter inverter", "electromagnetic compatibility EMI shielding",
        "antenna design gain directivity pattern", "microwave circuit stripline cavity",
        "VLSI design layout timing closure", "embedded system microcontroller firmware",
        "renewable energy solar photovoltaic", "electric motor drive torque speed",
        "battery management charge discharge cycle", "PCB printed circuit board layout",
        "analog-to-digital converter resolution", "PLL phase-locked loop frequency synthesis",
        "RF transceiver modulation link budget"
    ],
    "Civil Engineering": [
        "structural analysis beam column truss", "reinforced concrete design strength",
        "foundation soil bearing capacity pile", "earthquake resistant seismic isolation",
        "bridge span cable suspension arch", "highway pavement asphalt concrete design",
        "water supply distribution pipe network", "wastewater treatment plant design",
        "geotechnical slope stability retaining wall", "construction management schedule cost",
        "finite element structural modeling", "wind load building code provision",
        "prestressed concrete tendon anchor", "steel structure connection weld bolt",
        "traffic engineering signal intersection"
    ],
    "Mechanical Engineering": [
        "stress strain material yield fracture", "heat transfer conduction convection radiation",
        "fluid machinery pump turbine compressor", "vibration modal analysis damping",
        "gear transmission power train mechanism", "bearing lubrication friction wear",
        "thermodynamic cycle engine efficiency", "manufacturing CNC machining tolerance",
        "CAD solid modeling assembly design", "finite element stress thermal analysis",
        "mechanism linkage cam degree freedom", "HVAC heating ventilation air conditioning",
        "combustion engine fuel injection ignition", "pressure vessel boiler design code",
        "fatigue endurance limit reliability"
    ],
    "Aerospace Engineering": [
        "aerodynamics lift drag wing airfoil", "propulsion jet engine thrust turbine",
        "orbital mechanics satellite trajectory", "flight control autopilot stability",
        "structural lightweight composite airframe", "rocket propellant combustion nozzle",
        "reentry thermal protection ablation", "spacecraft attitude determination control",
        "mission planning trajectory optimization", "avionics navigation communication sensor",
        "wind tunnel testing model scale", "hypersonic flow shock heating",
        "satellite communication orbit constellation", "space debris collision avoidance",
        "launch vehicle staging payload fairing"
    ],
    "Biomedical Engineering": [
        "medical imaging MRI CT ultrasound", "prosthesis implant joint replacement",
        "tissue engineering scaffold regeneration", "biosensor diagnostic point-of-care",
        "drug delivery nanoparticle targeted release", "biomechanics gait motion analysis",
        "neural interface brain-computer electrode", "rehabilitation robotics exoskeleton",
        "clinical trial device regulation FDA", "bioinstrumentation physiological measurement",
        "artificial organ kidney heart lung", "surgical robot minimally invasive",
        "bioinformatics sequence protein structure", "stem cell therapy regenerative medicine",
        "wearable sensor health monitoring IoT"
    ],

    # DOMAIN 8: ECONOMICS & SOCIAL SCIENCE (16)
    "Microeconomics": [
        "supply demand equilibrium price quantity", "consumer utility maximization budget constraint",
        "firm production cost marginal revenue", "market structure monopoly oligopoly",
        "game theory Nash equilibrium strategy", "externality public good market failure",
        "price elasticity substitution income effect", "welfare economics Pareto efficiency",
        "auction mechanism design incentive", "labor market wage employment",
        "risk aversion expected utility insurance", "information asymmetry adverse selection moral",
        "behavioral economics prospect theory bias", "general equilibrium Arrow-Debreu",
        "industrial organization antitrust regulation"
    ],
    "Macroeconomics": [
        "GDP growth output national income", "inflation rate CPI monetary policy",
        "unemployment labor force participation", "fiscal policy government spending tax",
        "monetary policy interest rate central bank", "business cycle recession expansion",
        "international trade comparative advantage", "exchange rate currency devaluation",
        "aggregate demand supply IS-LM", "Phillips curve inflation unemployment tradeoff",
        "debt deficit sovereign bond yield", "economic growth Solow model capital",
        "expectation rational adaptive forward-looking", "financial crisis banking systemic risk",
        "development economics poverty inequality"
    ],
    "Finance": [
        "portfolio diversification risk return Sharpe", "option pricing Black-Scholes Greeks",
        "bond yield duration credit spread", "equity valuation DCF earnings ratio",
        "market efficiency EMH anomaly behavioral", "derivative futures forward swap",
        "capital structure leverage Modigliani-Miller", "risk management VaR stress test",
        "interest rate term structure yield curve", "banking regulation Basel capital adequacy",
        "corporate governance shareholder activist", "mergers acquisitions synergy premium",
        "behavioral finance bubble herding sentiment", "quantitative trading alpha signal",
        "fintech payment blockchain digital currency"
    ],
    "Political Science": [
        "democracy election voting system representation", "authoritarianism regime transition",
        "international relations diplomacy treaty alliance", "political party ideology coalition",
        "public policy implementation evaluation", "federalism decentralization devolution",
        "civil rights liberty constitution judiciary", "political economy inequality redistribution",
        "conflict war peace negotiation mediation", "nationalism identity state sovereignty",
        "globalization governance multilateral institution", "political behavior participation opinion",
        "comparative politics regime typology", "security studies deterrence defense",
        "environmental politics climate agreement governance"
    ],
    "Sociology": [
        "social stratification class inequality mobility", "institution norm value culture",
        "socialization identity role expectation", "deviance crime labeling control",
        "family marriage kinship household", "education achievement gap opportunity",
        "urbanization migration community neighborhood", "race ethnicity discrimination prejudice",
        "gender sexuality patriarchy feminism", "religion secularization fundamentalism",
        "social movement collective action protest", "bureaucracy organization power hierarchy",
        "media communication technology information", "health disparity social determinant",
        "globalization transnational network flow"
    ],
    "Psychology": [
        "cognition attention memory working", "emotion regulation appraisal coping",
        "personality trait Big Five model", "development stage Piaget attachment",
        "social influence conformity obedience", "motivation intrinsic extrinsic goal",
        "perception visual auditory illusion", "learning conditioning reinforcement",
        "psychopathology disorder diagnosis treatment", "intelligence IQ measurement factor",
        "stress resilience burnout wellbeing", "group dynamics leadership conflict",
        "decision making heuristic judgment bias", "consciousness awareness self-reflection",
        "therapy CBT psychoanalysis intervention"
    ],
    "Linguistics": [
        "phonology sound system phoneme allophone", "morphology word formation inflection derivation",
        "syntax sentence structure phrase grammar", "semantics meaning reference truth",
        "pragmatics context implicature speech act", "language acquisition first second critical",
        "sociolinguistics dialect variation register", "historical linguistics change reconstruction",
        "typology universal language diversity", "neurolinguistics brain Broca Wernicke",
        "computational linguistics parsing corpus", "phonetics articulation acoustic perception",
        "discourse analysis text coherence", "sign language gesture modality",
        "bilingualism code-switching multilingual"
    ],
    "Anthropology": [
        "ethnography fieldwork participant observation", "kinship descent marriage system",
        "ritual symbolism ceremony practice", "archaeological excavation artifact stratigraphy",
        "cultural relativism ethnocentrism critique", "human evolution fossil hominid",
        "language culture Sapir-Whorf hypothesis", "economic anthropology exchange reciprocity",
        "political anthropology power chiefdom state", "medical anthropology illness healing",
        "globalization cultural hybridity flow", "indigenous rights land sovereignty",
        "food agriculture subsistence production", "gender body performance identity",
        "visual anthropology film media representation"
    ],

    # DOMAIN 9: MEDICINE (15)
    "Cardiology": [
        "heart failure ejection fraction", "coronary artery disease atherosclerosis",
        "arrhythmia atrial fibrillation ablation", "myocardial infarction troponin",
        "echocardiography valve stenosis regurgitation", "hypertension blood pressure medication",
        "cardiac catheterization stent angioplasty", "congenital heart defect septal",
        "pericarditis effusion tamponade", "heart transplant ventricular assist device",
        "cardiac electrophysiology pacemaker ICD", "cardiomyopathy dilated hypertrophic",
        "pulmonary hypertension right ventricle", "atherosclerosis plaque lipid",
        "cardiac rehabilitation exercise training"
    ],
    "Oncology": [
        "tumor metastasis invasion angiogenesis", "chemotherapy cytotoxic drug resistance",
        "radiation therapy dose fractionation", "immunotherapy checkpoint inhibitor PD-1",
        "cancer staging TNM classification grade", "biopsy histopathology molecular marker",
        "breast cancer HER2 hormone receptor", "lung cancer NSCLC targeted therapy",
        "leukemia lymphoma bone marrow transplant", "colorectal cancer screening colonoscopy",
        "precision medicine genomic profiling mutation", "tumor microenvironment stroma immune",
        "palliative care quality life symptom", "carcinogenesis oncogene tumor suppressor",
        "liquid biopsy circulating tumor DNA"
    ],
    "Neurology": [
        "stroke ischemic hemorrhagic thrombolysis", "epilepsy seizure anticonvulsant EEG",
        "multiple sclerosis demyelination relapse", "Parkinson disease dopamine movement",
        "migraine headache aura trigger", "peripheral neuropathy nerve conduction",
        "traumatic brain injury concussion recovery", "spinal cord injury rehabilitation",
        "dementia cognitive decline vascular", "neuromuscular junction myasthenia gravis",
        "cerebrovascular aneurysm SAH intervention", "movement disorder dystonia tremor",
        "sleep disorder narcolepsy insomnia", "neuroimmunology encephalitis antibody",
        "neurogenetics Huntington SMA gene therapy"
    ],
    "Infectious Disease": [
        "viral infection pandemic respiratory", "bacterial sepsis antibiotic empiric",
        "HIV AIDS antiretroviral treatment", "tuberculosis multidrug resistant treatment",
        "malaria parasite antimalarial prophylaxis", "fungal infection immunocompromised",
        "vaccine development mRNA attenuated", "antimicrobial stewardship resistance",
        "nosocomial hospital-acquired infection control", "emerging infectious zoonotic spillover",
        "hepatitis viral liver cirrhosis", "influenza seasonal epidemic surveillance",
        "sexually transmitted infection screening", "travel medicine tropical endemic",
        "biofilm device-related chronic infection"
    ],
    "Orthopedics": [
        "fracture fixation plate screw nail", "joint replacement arthroplasty prosthesis",
        "sports injury ligament ACL meniscus", "spinal surgery fusion decompression",
        "osteoporosis bone density DEXA", "arthritis rheumatoid osteoarthritis treatment",
        "tendon repair rotator cuff Achilles", "bone tumor sarcoma resection",
        "pediatric orthopedic deformity scoliosis", "hand surgery carpal tunnel nerve",
        "shoulder dislocation instability repair", "knee arthroscopy cartilage restoration",
        "hip dysplasia osteotomy correction", "cast splint immobilization healing",
        "rehabilitation physiotherapy functional recovery"
    ],
    "Gastroenterology": [
        "inflammatory bowel disease Crohn ulcerative", "liver cirrhosis hepatitis fibrosis",
        "endoscopy colonoscopy biopsy polyp", "pancreatic disease cancer pancreatitis",
        "gastroesophageal reflux Barrett esophagus", "celiac disease gluten autoimmune",
        "irritable bowel syndrome functional disorder", "gallstone cholecystitis biliary",
        "gastrointestinal bleeding ulcer variceal", "colorectal cancer screening prevention",
        "microbiome gut-brain axis dysbiosis", "nutrition malabsorption enteral parenteral",
        "hepatocellular carcinoma surveillance treatment", "motility disorder gastroparesis achalasia",
        "fecal transplant Clostridium difficile"
    ],
    "Dermatology": [
        "melanoma skin cancer excision staging", "psoriasis plaque biologic therapy",
        "eczema atopic dermatitis barrier", "acne vulgaris treatment retinoid",
        "wound healing chronic ulcer debridement", "hair loss alopecia androgenetic",
        "autoimmune skin pemphigus bullous", "contact dermatitis allergen patch test",
        "skin infection cellulitis fungal viral", "cosmetic dermatology laser filler",
        "skin biopsy histopathology diagnosis", "phototherapy UVB PUVA vitiligo",
        "rosacea facial erythema trigger", "urticaria angioedema mast cell",
        "nail disorder onychomycosis dystrophy"
    ],
    "Ophthalmology": [
        "cataract surgery intraocular lens phaco", "glaucoma intraocular pressure optic nerve",
        "retinal detachment vitrectomy laser", "macular degeneration AMD anti-VEGF",
        "diabetic retinopathy screening laser", "refractive surgery LASIK cornea",
        "uveitis inflammation immunosuppression", "corneal transplant keratoplasty",
        "strabismus amblyopia pediatric", "dry eye syndrome tear film",
        "ocular trauma emergency management", "optic neuritis papilledema",
        "retinoblastoma pediatric tumor", "contact lens fitting complication",
        "visual field testing perimetry"
    ],
    "Psychiatry": [
        "depression major antidepressant SSRI", "anxiety disorder generalized panic",
        "schizophrenia psychosis antipsychotic dopamine", "bipolar disorder mood stabilizer lithium",
        "PTSD trauma exposure therapy EMDR", "OCD obsession compulsion ritual",
        "ADHD attention deficit stimulant medication", "eating disorder anorexia bulimia",
        "substance abuse addiction withdrawal detox", "personality disorder borderline antisocial",
        "suicide risk assessment prevention intervention", "autism spectrum disorder communication",
        "insomnia sleep pharmacotherapy CBT-I", "dementia behavioral neuropsychiatric symptom",
        "electroconvulsive therapy ECT severe depression"
    ],
    "Radiology": [
        "CT computed tomography contrast scan", "MRI magnetic resonance imaging sequence",
        "ultrasound Doppler echocardiography obstetric", "X-ray radiograph chest bone",
        "PET nuclear medicine FDG oncology", "interventional radiology catheter embolization",
        "mammography breast imaging screening", "fluoroscopy real-time guided procedure",
        "contrast agent gadolinium iodine reaction", "radiation dose ALARA optimization",
        "image-guided biopsy drainage ablation", "musculoskeletal imaging joint tendon",
        "neuroradiology brain spine stroke", "chest imaging pulmonary nodule",
        "abdominal imaging liver kidney pancreas"
    ],
    "Pharmacology": [
        "drug receptor binding agonist antagonist", "pharmacokinetics absorption distribution metabolism",
        "drug interaction cytochrome P450", "adverse drug reaction toxicity monitoring",
        "dose-response curve therapeutic window", "clinical trial phase randomized controlled",
        "drug development target validation lead", "generic biosimilar bioequivalence",
        "opioid analgesic pain management", "anticoagulant warfarin heparin DOAC",
        "antibiotic mechanism spectrum coverage", "cardiovascular drug statin ACE inhibitor",
        "psychopharmacology antidepressant anxiolytic", "pharmacogenomics personalized medicine",
        "drug delivery formulation sustained release"
    ],
    "Emergency Medicine": [
        "trauma resuscitation ATLS primary survey", "cardiac arrest CPR defibrillation ACLS",
        "airway management intubation ventilation", "shock septic hypovolemic distributive",
        "fracture dislocation reduction splinting", "burn assessment fluid resuscitation",
        "poisoning toxicology antidote decontamination", "anaphylaxis epinephrine emergency",
        "stroke code thrombolysis time window", "acute abdomen surgical emergency",
        "pediatric emergency fever seizure dehydration", "chest pain triage risk stratification",
        "wound closure suture laceration repair", "procedural sedation analgesia",
        "disaster triage mass casualty incident"
    ],
    "Surgery": [
        "laparoscopic minimally invasive technique", "anesthesia general regional local",
        "wound infection surgical site prevention", "hernia repair mesh laparoscopic",
        "appendectomy cholecystectomy acute abdomen", "bariatric surgery obesity gastric bypass",
        "vascular surgery bypass graft aneurysm", "thoracic surgery lung resection",
        "transplant organ procurement immunosuppression", "plastic reconstructive flap graft",
        "surgical oncology resection margin lymph", "perioperative care enhanced recovery",
        "robotic surgery da Vinci precision", "hemostasis bleeding coagulation",
        "postoperative complication DVT ileus"
    ],
    "Endocrinology": [
        "diabetes mellitus insulin glucose HbA1c", "thyroid disorder hypothyroid hyperthyroid",
        "adrenal Cushing Addison cortisol", "pituitary adenoma prolactin growth hormone",
        "osteoporosis calcium vitamin D bone", "polycystic ovary syndrome PCOS",
        "metabolic syndrome insulin resistance obesity", "testosterone hypogonadism replacement",
        "parathyroid calcium hyperparathyroidism", "neuroendocrine tumor carcinoid",
        "diabetic complication nephropathy neuropathy retinopathy", "obesity weight management bariatric",
        "adrenal insufficiency crisis steroid", "growth disorder short stature pediatric",
        "thyroid nodule cancer differentiated"
    ],
    "Pulmonology": [
        "asthma bronchospasm inhaler corticosteroid", "COPD emphysema chronic bronchitis",
        "pneumonia community hospital aspiration", "pulmonary embolism DVT anticoagulation",
        "interstitial lung disease fibrosis", "lung cancer staging surgery chemo",
        "tuberculosis latent active treatment", "sleep apnea CPAP polysomnography",
        "pleural effusion thoracentesis drainage", "bronchiectasis mucus clearance infection",
        "pulmonary function test spirometry DLCO", "mechanical ventilation ARDS PEEP",
        "sarcoidosis granuloma multisystem", "pulmonary hypertension right heart",
        "cystic fibrosis CFTR mutation therapy"
    ],

    # DOMAIN 10: HISTORY & PHILOSOPHY (15)
    "Ancient History": [
        "Roman Empire republic senate emperor", "Greek civilization democracy philosophy",
        "Egyptian pyramid pharaoh hieroglyphic", "Mesopotamia Sumerian cuneiform city-state",
        "Persian Empire Achaemenid Darius expansion", "Han Dynasty Silk Road dynasty",
        "Maurya Gupta Empire Indian subcontinent", "Phoenician trade Mediterranean colony",
        "classical archaeology excavation artifact", "ancient warfare legion phalanx tactics",
        "slavery institution ancient economy labor", "Roman law codification jurisprudence",
        "Hellenistic period Alexander successor kingdom", "ancient religion polytheism temple ritual",
        "ancient technology aqueduct road engineering"
    ],
    "Medieval History": [
        "feudalism lord vassal manor estate", "Crusades Holy Land Jerusalem military",
        "Black Death plague pandemic medieval", "medieval church papacy monastery",
        "Viking Norse exploration raid settlement", "Byzantine Empire Constantinople Orthodox",
        "Islamic Golden Age caliphate scholarship", "Hundred Years War England France",
        "medieval trade guild merchant Hanseatic", "castle fortification siege warfare",
        "Magna Carta constitutional monarchy Parliament", "scholasticism university theology Aquinas",
        "chivalry knight tournament heraldry", "agricultural revolution three-field plow",
        "Mongol Empire conquest Genghis Khan"
    ],
    "Modern History": [
        "Industrial Revolution factory urbanization labor", "French Revolution Enlightenment republic",
        "World War trench alliance total war", "Cold War superpower nuclear deterrence",
        "colonialism imperialism decolonization independence", "American Revolution constitution republic",
        "nationalism unification self-determination", "Russian Revolution Bolshevik Soviet",
        "Great Depression economic crisis recovery", "civil rights movement equality segregation",
        "space race exploration Apollo moon", "globalization trade technology migration",
        "nuclear age arms race proliferation", "decolonization Africa Asia independence",
        "European Union integration community market"
    ],
    "Philosophy of Mind": [
        "consciousness qualia hard problem", "dualism monism mind body problem",
        "intentionality aboutness mental content", "free will determinism compatibilism",
        "personal identity continuity memory", "mental causation epiphenomenalism",
        "functionalism computational theory mind", "eliminative materialism folk psychology",
        "phenomenology lived experience Husserl", "artificial intelligence Chinese room Turing",
        "embodied cognition extended mind", "perception representation direct realism",
        "emotion cognition feeling affect", "self-awareness introspection metacognition",
        "philosophy cognitive science neuroscience"
    ],
    "Ethics": [
        "utilitarianism consequentialism greatest good", "deontology Kantian duty categorical",
        "virtue ethics Aristotle character flourishing", "moral relativism cultural universal",
        "bioethics medical autonomy consent", "environmental ethics value nature",
        "social justice fairness Rawls veil", "animal rights welfare suffering",
        "applied ethics professional business", "metaethics moral realism naturalism",
        "care ethics feminist relational", "consequentialism act rule indirect",
        "moral responsibility agency blame praise", "human rights dignity inalienable",
        "technology ethics AI privacy surveillance"
    ],
    "Epistemology": [
        "knowledge justified true belief Gettier", "skepticism doubt certainty Descartes",
        "empiricism rationalism a priori posteriori", "foundationalism coherentism epistemic",
        "truth correspondence coherence pragmatic", "perception evidence testimony",
        "induction problem Hume uniformity", "scientific method hypothesis falsification",
        "social epistemology testimony trust", "Bayesian reasoning belief updating",
        "reliabilism process externalism internalism", "virtue epistemology intellectual",
        "contextualism standards knowledge attribution", "pragmatism James Dewey inquiry",
        "naturalized epistemology Quine cognitive"
    ],
    "Logic": [
        "propositional calculus truth table connective", "predicate logic quantifier variable",
        "modal logic necessity possibility world", "proof theory deduction natural sequent",
        "model theory interpretation satisfaction", "set theory axiom Zermelo-Fraenkel",
        "Gödel incompleteness theorem undecidable", "computability recursion Turing halting",
        "intuitionistic logic constructive proof", "paraconsistent logic contradiction tolerance",
        "fuzzy logic degree truth membership", "temporal logic time LTL CTL",
        "deontic logic obligation permission", "non-monotonic reasoning default",
        "type theory lambda calculus Curry-Howard"
    ],
    "Aesthetics": [
        "beauty sublime judgment taste", "art definition expression representation",
        "aesthetic experience contemplation pleasure", "music philosophy form meaning emotion",
        "literary criticism interpretation hermeneutics", "film theory montage narrative",
        "architecture space form function aesthetic", "dance performance body movement",
        "creativity imagination artistic process", "cultural value canon tradition",
        "tragedy comedy dramatic genre", "mimesis imitation reality art",
        "aesthetic autonomy art for art sake", "postmodernism deconstruction irony pastiche",
        "environmental aesthetics landscape nature"
    ],
    "Philosophy of Science": [
        "scientific realism anti-realism entity", "paradigm shift Kuhn revolution normal",
        "falsificationism Popper demarcation criterion", "underdetermination theory evidence",
        "explanation causation mechanism law", "observation theory-laden neutral data",
        "reduction emergence levels composition", "natural kind classification taxonomy",
        "probability confirmation Bayesian induction", "scientific model idealization abstraction",
        "experiment intervention control variable", "scientific progress accumulation revolution",
        "values objectivity science society", "measurement operationalism construct",
        "unification explanatory power scope"
    ],
    "Political Philosophy": [
        "social contract Hobbes Locke Rousseau", "justice distributive Rawls difference principle",
        "liberty positive negative freedom Berlin", "democracy deliberative participation legitimacy",
        "authority obligation consent political", "rights natural human civil property",
        "equality opportunity outcome sufficiency", "communitarianism individual community virtue",
        "anarchism state abolition voluntary", "feminism patriarchy gender justice",
        "marxism class exploitation alienation", "liberalism tolerance pluralism neutrality",
        "conservatism tradition order authority", "cosmopolitanism global justice borders",
        "sovereignty self-determination nationalism territory"
    ],

    # DOMAIN 11: ARTS & HUMANITIES (14)
    "Music Theory": [
        "harmony chord progression voice leading", "rhythm meter tempo syncopation",
        "melody interval scale mode", "counterpoint fugue canon polyphony",
        "orchestration timbre instrumentation arrangement", "form sonata rondo theme variation",
        "tonality key modulation chromatic", "dynamics articulation phrasing expression",
        "atonal serialism twelve-tone Schoenberg", "acoustics overtone resonance frequency",
        "notation score manuscript tablature", "improvisation jazz bebop modal",
        "electronic music synthesis sampling", "ethnomusicology world music tradition",
        "composition technique development recapitulation"
    ],
    "Visual Arts": [
        "painting oil acrylic watercolor canvas", "sculpture marble bronze casting carving",
        "photography composition exposure aperture", "printmaking etching lithography woodcut",
        "drawing charcoal graphite perspective shading", "color theory hue saturation value",
        "abstract expressionism action painting gesture", "impressionism light plein-air brushwork",
        "Renaissance perspective anatomy proportion", "contemporary installation performance conceptual",
        "digital art generative interactive media", "ceramic pottery glaze kiln",
        "textile weaving fiber pattern", "mosaic fresco mural wall",
        "gallery exhibition curation collection"
    ],
    "Film Studies": [
        "cinematography camera angle shot framing", "editing montage continuity jump cut",
        "narrative structure plot character arc", "mise-en-scène set design costume",
        "sound design dialogue score Foley", "genre convention horror comedy thriller",
        "documentary realism verité observational", "animation frame hand-drawn CGI",
        "auteur director vision personal style", "film theory spectatorship gaze apparatus",
        "screenplay writing dialogue scene structure", "production budget schedule casting",
        "post-production color grading VFX", "distribution exhibition streaming festival",
        "film history silent era studio system"
    ],
    "Theater": [
        "dramatic text script playwright dialogue", "acting method Stanislavski technique emotion",
        "directing staging blocking interpretation", "set design scenic construction space",
        "costume design wardrobe character visual", "lighting design cue atmosphere mood",
        "sound design underscore effect environment", "musical theater song dance book",
        "improvisation devising ensemble creation", "audience reception participation immersive",
        "Greek tragedy chorus catharsis mask", "Shakespeare Elizabethan verse soliloquy",
        "absurdist theater Beckett existential", "political theater Brecht epic alienation",
        "puppetry mask physical theater movement"
    ],
    "Literature": [
        "novel narrative fiction character plot", "poetry verse meter rhyme imagery",
        "drama play performance stage dialogue", "short story flash fiction narrative",
        "literary theory criticism interpretation reading", "modernism stream consciousness experimentation",
        "postcolonial literature voice identity subaltern", "gothic horror supernatural uncanny",
        "science fiction speculative dystopia utopia", "realism naturalism social observation",
        "magical realism ordinary extraordinary", "autobiography memoir personal narrative",
        "satire irony parody social critique", "epic mythology heroic journey",
        "translation adaptation cultural text"
    ],
    "Architecture": [
        "structural system load bearing frame", "building material concrete steel timber",
        "sustainable design green building LEED", "urban planning zoning density mixed-use",
        "facade envelope cladding thermal insulation", "interior space program circulation",
        "architectural history modernism postmodern", "parametric design computation generative",
        "vernacular regional context climate responsive", "restoration preservation heritage conservation",
        "landscape architecture garden public space", "acoustics thermal lighting comfort",
        "high-rise tower skyscraper foundation", "housing typology apartment collective",
        "infrastructure bridge tunnel station"
    ],
    "Dance": [
        "ballet classical technique pointe barre", "contemporary modern release improvisation",
        "choreography composition structure phrase", "movement analysis Laban effort shape",
        "performance stage presence audience", "rehearsal process collaboration company",
        "body conditioning flexibility strength alignment", "rhythm musicality timing accent",
        "cultural dance folk traditional ritual", "hip-hop street breaking popping",
        "notation Labanotation Benesh documentation", "dance film screen adaptation",
        "somatic practice Feldenkrais Alexander", "contact improvisation partner weight",
        "dance education pedagogy training curriculum"
    ],
    "Creative Writing": [
        "narrative voice point-of-view perspective", "character development motivation arc",
        "setting world-building atmosphere detail", "dialogue subtext conflict tension",
        "plot structure rising action climax", "revision editing workshop feedback",
        "genre fiction mystery romance thriller", "poetry form free verse sonnet",
        "memoir personal essay reflection", "screenplay format scene beat",
        "flash fiction compression economy image", "publishing submission agent query",
        "style tone register diction syntax", "imagery metaphor simile symbolism",
        "theme universal particular meaning"
    ],

    # DOMAIN 12: ENVIRONMENTAL & ENERGY (12)
    "Renewable Energy": [
        "solar photovoltaic panel efficiency cell", "wind turbine blade generator capacity",
        "hydropower dam turbine reservoir flow", "geothermal heat pump deep well",
        "biomass biofuel ethanol biodiesel feedstock", "tidal wave ocean energy converter",
        "energy storage battery grid lithium", "smart grid demand response management",
        "hydrogen fuel cell electrolysis green", "concentrated solar power thermal tower",
        "offshore wind floating foundation cable", "energy policy subsidy tariff market",
        "grid integration intermittent variable", "energy efficiency building retrofit",
        "microgrid island distributed generation"
    ],
    "Climate Science": [
        "global warming temperature anomaly trend", "greenhouse effect radiative forcing CO2",
        "ice sheet melting sea level rise", "ocean circulation thermohaline AMOC",
        "extreme weather attribution frequency intensity", "carbon cycle sink source flux",
        "climate model projection scenario RCP", "paleoclimate proxy ice core sediment",
        "aerosol cloud feedback sensitivity", "permafrost thaw methane emission",
        "tipping point irreversible threshold", "mitigation adaptation resilience policy",
        "carbon budget remaining emissions pathway", "Arctic amplification albedo feedback",
        "climate justice equity vulnerable developing"
    ],
    "Conservation Biology": [
        "endangered species red list extinction", "habitat loss fragmentation degradation",
        "protected area reserve design connectivity", "wildlife management population monitoring",
        "biodiversity hotspot endemic species priority", "restoration ecology rewilding native",
        "invasive species control eradication management", "captive breeding reintroduction program",
        "ecosystem service valuation natural capital", "genetic diversity bottleneck inbreeding",
        "marine conservation MPA fishing bycatch", "deforestation land-use change tropical",
        "poaching trafficking illegal wildlife trade", "citizen science community monitoring",
        "conservation policy CITES convention agreement"
    ],
    "Pollution Control": [
        "air quality emission standard regulation", "water pollution treatment effluent discharge",
        "soil remediation contamination cleanup", "hazardous waste management disposal",
        "noise pollution mitigation barrier standard", "industrial emission scrubber filter",
        "vehicle exhaust catalytic converter", "particulate matter PM health respiratory",
        "heavy metal contamination bioaccumulation", "oil spill response containment cleanup",
        "plastic waste recycling circular economy", "thermal pollution cooling discharge aquatic",
        "light pollution ecology circadian sky", "electromagnetic radiation exposure limit",
        "environmental monitoring sensor network"
    ],
    "Ecology & Ecosystems": [
        "forest ecosystem canopy understory decomposition", "wetland marsh swamp bog function",
        "grassland savanna prairie fire regime", "desert arid adaptation xerophyte",
        "freshwater lake river stream ecosystem", "estuary brackish tidal mixing",
        "tundra permafrost arctic alpine", "coral reef tropical marine diversity",
        "mangrove coastal protection nursery", "deep ocean abyssal pelagic benthic",
        "urban ecology city green space wildlife", "agricultural ecosystem crop rotation soil",
        "boreal taiga conifer winter adaptation", "island ecology isolation endemism",
        "cave karst subterranean darkness adapted"
    ],
    "Waste Management": [
        "recycling material recovery sorting", "landfill design liner leachate gas",
        "incineration waste-to-energy emission", "composting organic biodegradable aerobic",
        "electronic waste e-waste precious metal", "nuclear waste high-level repository",
        "sewage wastewater treatment plant sludge", "construction demolition debris reuse",
        "biomedical waste infectious sharps autoclave", "industrial waste minimization prevention",
        "circular economy product lifecycle design", "extended producer responsibility packaging",
        "ocean plastic debris microplastic marine", "food waste reduction donation compost",
        "hazardous chemical storage transport disposal"
    ],
    "Water Resources": [
        "freshwater scarcity supply demand", "desalination membrane reverse osmosis",
        "water quality testing standard parameter", "watershed management integrated planning",
        "groundwater depletion over-extraction aquifer", "irrigation efficiency drip sprinkler",
        "flood control levee dam spillway", "water pricing allocation rights market",
        "rainwater harvesting collection storage", "transboundary water conflict cooperation",
        "drinking water treatment filtration disinfection", "water reuse recycled wastewater",
        "river restoration channel habitat fish", "wetland constructed treatment natural",
        "drought management contingency planning"
    ],
    "Sustainable Agriculture": [
        "organic farming certification pesticide-free", "soil health carbon sequestration tillage",
        "integrated pest management biological control", "crop rotation diversification cover crop",
        "precision agriculture GPS sensor variable-rate", "agroforestry tree crop integration shade",
        "water-efficient irrigation scheduling deficit", "pollinator conservation bee habitat",
        "food security nutrition access availability", "seed saving genetic resource heirloom",
        "composting manure nutrient cycling", "livestock welfare pasture grazing rotation",
        "urban farming vertical indoor hydroponic", "climate-smart agriculture adaptation mitigation",
        "fair trade certification smallholder income"
    ],

    # DOMAIN 13: TECHNOLOGY & INDUSTRY (14)
    "Artificial Intelligence": [
        "large language model GPT transformer", "reasoning planning agent goal-directed",
        "knowledge representation ontology graph", "expert system rule-based inference",
        "multi-agent system coordination negotiation", "AI safety alignment value specification",
        "explainable interpretable black-box model", "few-shot learning prompt engineering",
        "autonomous vehicle self-driving perception", "recommendation system collaborative filtering",
        "AI ethics bias fairness accountability", "generative AI image text synthesis",
        "AI regulation governance framework policy", "embodied AI robot interaction",
        "artificial general intelligence capability benchmark"
    ],
    "Cybersecurity": [
        "malware virus ransomware trojan", "network intrusion detection prevention IDS",
        "vulnerability exploit zero-day patch", "authentication multi-factor identity access",
        "encryption data protection privacy", "phishing social engineering awareness",
        "penetration testing ethical hacking assessment", "incident response forensics recovery",
        "firewall access control policy rule", "secure software development SDLC",
        "cloud security shared responsibility model", "IoT security device firmware update",
        "supply chain attack third-party risk", "threat intelligence indicator compromise",
        "compliance regulation GDPR HIPAA audit"
    ],
    "Cloud Computing": [
        "virtual machine container orchestration Kubernetes", "serverless function-as-a-service event",
        "infrastructure-as-code Terraform provisioning", "microservice API gateway service mesh",
        "auto-scaling elasticity load balancer", "object storage blob bucket S3",
        "database-as-a-service managed instance", "CI/CD pipeline deployment automation",
        "multi-cloud hybrid edge computing", "identity access management IAM role",
        "cost optimization reserved spot instance", "monitoring observability logging metrics",
        "disaster recovery backup replication", "network virtual private cloud VPC subnet",
        "compliance certification SOC ISO region"
    ],
    "Internet of Things": [
        "sensor actuator embedded microcontroller", "wireless protocol BLE Zigbee LoRa",
        "edge computing fog gateway processing", "smart home automation device hub",
        "industrial IoT predictive maintenance", "wearable health fitness tracker",
        "MQTT CoAP messaging lightweight protocol", "digital twin simulation real-time",
        "energy harvesting battery-less self-powered", "mesh network topology connectivity",
        "IoT platform data ingestion analytics", "smart city infrastructure traffic",
        "agriculture precision monitoring irrigation", "supply chain tracking RFID NFC",
        "security firmware OTA update"
    ],
    "Semiconductor Technology": [
        "transistor CMOS scaling Moore law", "fabrication lithography etching deposition",
        "chip design RTL synthesis layout", "memory DRAM SRAM flash NAND",
        "processor architecture pipeline cache", "packaging interconnect 3D stacking",
        "power management voltage regulator", "analog mixed-signal ADC DAC",
        "testing yield defect reliability", "EDA electronic design automation tool",
        "quantum computing qubit superconducting", "photonic integrated circuit optical",
        "MEMS microelectromechanical sensor actuator", "wide-bandgap GaN SiC power",
        "neuromorphic computing spiking memristor"
    ],
    "Telecommunications": [
        "5G NR millimeter wave beamforming", "fiber optic WDM amplifier long-haul",
        "satellite communication LEO constellation", "spectrum allocation frequency band license",
        "antenna MIMO massive array", "modulation OFDM QAM constellation",
        "network slicing virtualization NFV SDN", "channel coding LDPC polar turbo",
        "handover mobility management roaming", "backhaul fronthaul transport network",
        "interference management power control", "IoT connectivity LPWAN NB-IoT",
        "optical networking DWDM coherent", "radio access network RAN base station",
        "quality experience QoE latency jitter"
    ],
    "Automotive Technology": [
        "electric vehicle battery powertrain", "autonomous driving sensor fusion ADAS",
        "internal combustion engine efficiency emission", "transmission gearbox CVT dual-clutch",
        "vehicle dynamics handling stability suspension", "safety crash test airbag structure",
        "aerodynamics drag coefficient design", "connected vehicle V2X communication",
        "fuel cell hydrogen electric drive", "manufacturing assembly line quality",
        "infotainment display interface HMI", "tire traction grip rolling resistance",
        "hybrid powertrain regenerative braking", "chassis platform architecture modular",
        "emission standard regulation Euro EPA"
    ],
    "Manufacturing": [
        "CNC machining precision tolerance surface", "injection molding plastic tooling cycle",
        "additive manufacturing 3D printing metal", "welding joining brazing adhesive",
        "quality control inspection SPC six-sigma", "lean manufacturing waste elimination Kaizen",
        "supply chain logistics inventory JIT", "automation PLC SCADA industrial control",
        "casting forging forming sheet metal", "surface treatment coating anodizing plating",
        "metrology measurement CMM gauge calibration", "tool wear cutting speed feed",
        "assembly line ergonomics efficiency layout", "maintenance predictive preventive TPM",
        "Industry 4.0 digital factory twin"
    ],
    "Nanotechnology": [
        "nanoparticle synthesis functionalization", "carbon nanotube graphene 2D material",
        "quantum dot fluorescence semiconductor", "self-assembly supramolecular bottom-up",
        "nanomedicine drug delivery targeting", "nano-fabrication lithography e-beam",
        "surface plasmon resonance sensing", "nanocomposite polymer clay fiber",
        "scanning probe microscopy AFM STM", "nanoelectronics molecular transistor",
        "nano-optics photonic crystal metamaterial", "toxicology environmental nano safety",
        "thin film deposition ALD MBE", "nanowire growth semiconductor", "DNA nanotechnology origami"
    ],
    "Food Science": [
        "food preservation pasteurization sterilization", "fermentation yeast bacteria cheese yogurt",
        "food safety HACCP pathogen contamination", "nutritional analysis macronutrient vitamin",
        "food processing extrusion drying freezing", "flavor chemistry volatile compound taste",
        "food packaging shelf-life barrier material", "texture rheology emulsion gel foam",
        "food additive preservative colorant sweetener", "dairy science milk protein casein",
        "meat science muscle protein quality", "cereal grain starch gluten baking",
        "food engineering heat mass transfer", "sensory evaluation panel tasting",
        "functional food nutraceutical health claim"
    ],
    "Textile Technology": [
        "fiber spinning natural synthetic polymer", "yarn formation twisting plying",
        "weaving loom warp weft pattern", "knitting weft warp machine",
        "dyeing finishing color fastness process", "textile testing strength elongation",
        "nonwoven fabric needle-punch meltblown", "smart textile sensor conductive",
        "composite technical textile reinforcement", "garment construction pattern sewing",
        "sustainability recycling circular fashion", "water repellent flame retardant finish",
        "printing digital screen rotary", "quality control defect inspection",
        "fiber identification microscopy chemical test"
    ],
    "Mining Engineering": [
        "ore deposit geological exploration reserve", "open pit underground extraction method",
        "blasting drilling fragmentation rock", "mineral processing crushing grinding flotation",
        "mine ventilation safety gas dust", "tailings dam waste management environmental",
        "rock mechanics stability support reinforcement", "mine planning scheduling optimization",
        "dewatering pumping groundwater control", "coal mining longwall room pillar",
        "gold copper iron extraction metallurgy", "mine closure rehabilitation reclamation",
        "geotechnical investigation borehole logging", "surveying mapping GPS monitoring",
        "automation remote operation autonomous equipment"
    ],
}

# Templates for document generation
TEMPLATES = [
    "Research on {kw1} has revealed important connections to {kw2}. Recent findings about {kw3} provide new insights.",
    "The study of {kw1} requires understanding {kw2}. Furthermore, {kw3} plays a crucial role in this field.",
    "Analysis of {kw1} demonstrates significant overlap with {kw2}. The implications for {kw3} are substantial.",
    "Investigating {kw1} alongside {kw2} has opened new avenues. The relationship to {kw3} warrants further study.",
    "Recent advances in {kw1} complement work on {kw2}. Together with {kw3}, these form a coherent framework.",
    "The intersection of {kw1} and {kw2} presents unique challenges. Progress in {kw3} may provide solutions.",
    "Understanding {kw1} is essential for progress in {kw2}. Applications involving {kw3} are particularly promising.",
    "Combining insights from {kw1} with {kw2} has led to breakthroughs. The role of {kw3} cannot be overlooked.",
    "A comprehensive review of {kw1} highlights the importance of {kw2}. New developments in {kw3} are noteworthy.",
    "Experimental evidence for {kw1} supports theories about {kw2}. The connection to {kw3} strengthens this conclusion.",
]


def generate_documents(n_per_category=100, seed=42):
    """Generate text documents from category keywords."""
    random.seed(seed)
    np.random.seed(seed)
    
    documents = []
    category_names = list(CATEGORIES.keys())
    
    for cat_idx, cat_name in enumerate(category_names):
        keywords = CATEGORIES[cat_name]
        
        for doc_i in range(n_per_category):
            template = random.choice(TEMPLATES)
            
            # Pick 3 keywords from this category
            kws = random.sample(keywords, min(3, len(keywords)))
            kw_dict = {f'kw{i+1}': kws[i] for i in range(3)}
            text = template.format(**kw_dict)
            
            documents.append({
                'text': text,
                'category_idx': cat_idx,
                'category_name': cat_name,
            })
    
    random.shuffle(documents)
    return documents, category_names


def build_features(documents, category_names, d=200, seed=42, **kwargs):
    """
    Build BoW features and project to d dimensions.
    
    Returns:
      doc_features: (N, d) unit-norm projected document features
      arm_features: (K, d) unit-norm projected arm features
    """
    np.random.seed(seed)
    
    # Build vocabulary from all keywords
    all_keywords = []
    keyword_to_idx = {}
    for cat_name, kws in CATEGORIES.items():
        for kw in kws:
            if kw not in keyword_to_idx:
                keyword_to_idx[kw] = len(all_keywords)
                all_keywords.append(kw)
    
    V = len(all_keywords)
    N = len(documents)
    K = len(category_names)
    
    print(f"  Vocabulary size: {V}")
    print(f"  N docs: {N}, K arms: {K}")
    
    # Document BoW
    doc_bow = np.zeros((N, V))
    for i, doc in enumerate(documents):
        text_lower = doc['text'].lower()
        for kw, idx in keyword_to_idx.items():
            if kw.lower() in text_lower:
                doc_bow[i, idx] = 1.0
    
    # Arm BoW (each arm = its category's keyword vector)
    arm_bow = np.zeros((K, V))
    for k, cat_name in enumerate(category_names):
        for kw in CATEGORIES[cat_name]:
            if kw in keyword_to_idx:
                arm_bow[k, keyword_to_idx[kw]] = 1.0
    
    # Random projection to d dimensions
    W = np.random.randn(V, d) / np.sqrt(d)
    
    doc_proj = doc_bow @ W
    arm_proj = arm_bow @ W
    
    # Add noise to document features to degrade LinUCB
    # (LLM reads raw text, unaffected by feature noise)
    noise_std = kwargs.get('noise_std', 2.0)
    if noise_std > 0:
        doc_proj += noise_std * np.random.randn(*doc_proj.shape)
    
    # Normalize to unit sphere
    doc_norms = norm(doc_proj, axis=1, keepdims=True)
    doc_norms[doc_norms < 1e-8] = 1.0
    doc_features = doc_proj / doc_norms
    
    arm_norms = norm(arm_proj, axis=1, keepdims=True)
    arm_norms[arm_norms < 1e-8] = 1.0
    arm_features = arm_proj / arm_norms
    
    # Diagnostics
    # Coherence: average |<a_i, a_j>| for i≠j
    n_pairs = min(5000, K * (K-1) // 2)
    coherences = []
    for _ in range(n_pairs):
        i, j = np.random.choice(K, 2, replace=False)
        coherences.append(abs(arm_features[i] @ arm_features[j]))
    alpha_hat = np.mean(coherences)
    
    # Effective rank of φ(x,a) = x ⊙ a covariance
    sample_size = min(5000, N)
    sample_idx = np.random.choice(N, sample_size, replace=False)
    sample_arms = np.array([documents[i]['category_idx'] for i in sample_idx])
    phi_samples = doc_features[sample_idx] * arm_features[sample_arms]
    cov = phi_samples.T @ phi_samples / sample_size
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)
    
    print(f"  Coherence α̂: {alpha_hat:.4f}")
    print(f"  Effective rank r̂: {r_hat:.1f}")
    print(f"  Feature dim d: {d}")
    
    return doc_features.astype(np.float32), arm_features.astype(np.float32), {
        'alpha_hat': float(alpha_hat),
        'r_hat': float(r_hat),
        'd': d, 'K': K, 'N': N, 'V': V,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_per_cat', type=int, default=100,
                        help='Documents per category')
    parser.add_argument('--d', type=int, default=200,
                        help='Feature dimension after projection')
    parser.add_argument('--noise', type=float, default=2.0,
                        help='Noise std added to doc features (degrades LinUCB)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='synthetic_text_dataset_v8.pkl.gz')
    args = parser.parse_args()
    
    K = len(CATEGORIES)
    print(f"=" * 70)
    print(f"SYNTHETIC TEXT DATASET v8")
    print(f"  {K} categories, {args.n_per_cat} docs/cat = {K * args.n_per_cat} total")
    print(f"  Feature dim: {args.d}")
    print(f"=" * 70)
    
    # Generate documents
    print("\n[1] Generating documents...")
    documents, category_names = generate_documents(
        n_per_category=args.n_per_cat, seed=args.seed
    )
    print(f"  Generated {len(documents)} documents across {len(category_names)} categories")
    
    # Build features
    print("\n[2] Building features...")
    doc_features, arm_features, diagnostics = build_features(
        documents, category_names, d=args.d, seed=args.seed, noise_std=args.noise
    )
    
    # Assemble dataset
    ys = np.array([doc['category_idx'] for doc in documents])
    texts = [doc['text'] for doc in documents]
    
    dataset = {
        'doc_features': doc_features,
        'arm_features': arm_features,
        'ys': ys,
        'texts': texts,
        'category_names': category_names,
        'diagnostics': diagnostics,
        'n_per_cat': args.n_per_cat,
    }
    
    # Save
    print(f"\n[3] Saving to {args.out}...")
    with gzip.open(args.out, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDone! Dataset: {K} categories, {len(documents)} docs, d={args.d}")
    print(f"Diagnostics: α̂={diagnostics['alpha_hat']:.4f}, r̂={diagnostics['r_hat']:.1f}")
    print(f"\nNext step: run LLM inference with run_llm_inference_v8.py")


if __name__ == '__main__':
    main()
