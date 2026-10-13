"""
generate_dataset_v3.py

Final synthetic text dataset that achieves warm-start-essential regime.

Key design:
- 50 categories with 30 keywords each (20 unique + 10 shared with other categories)
- Bag-of-words document features → PCA projection to d=200 (for tractable LinUCB)
- Keyword overlap creates genuine confusion → heavy gap tail (Axis C)
- Disjoint unique keywords → low coherence (Axis B)
- High effective rank in projected space → large r̂/√T (Axis A)
- T=2000 experiment horizon (where Axis A deviates)
- 2000+ text documents, with raw text available for LLM classification

The pipeline:
- LinUCB uses projected BoW features (d=200)
- LLM reads raw text and classifies
- Router (Algorithm 1) decides between them
"""

import numpy as np
import random
import gzip
import pickle
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# Reuse the rich category keywords from v2
CATEGORIES = {
    # PHYSICAL SCIENCES (10)
    "Quantum Mechanics": [
        "wave function collapse", "Heisenberg uncertainty", "quantum entanglement",
        "superposition state", "Schrödinger equation", "photon polarization",
        "tunneling barrier", "Planck constant", "spin angular momentum",
        "observable eigenvalue", "quantum decoherence", "Hilbert space",
        "probability amplitude", "interference pattern", "Bohr model",
        "quantum field", "vacuum fluctuation", "path integral",
        "density matrix", "Bell inequality", "EPR paradox",
        "quantum teleportation", "Pauli exclusion", "Dirac notation",
        "zero-point energy", "Bose-Einstein condensate", "wavepacket",
        "measurement problem", "many-worlds interpretation", "quantum chromodynamics"
    ],
    "Volcanic Activity": [
        "magma chamber pressure", "pyroclastic flow", "eruption plume",
        "lava dome growth", "tectonic subduction", "volcanic tremor",
        "caldera collapse", "pumice ejection", "geothermal vent",
        "sulfur dioxide emission", "obsidian formation", "stratovolcano",
        "shield volcano basalt", "phreatic explosion", "lahar mudflow",
        "volcanic bomb", "fumarole gas", "ignimbrite deposit",
        "mantle plume theory", "volcanic arc formation", "tephra layer",
        "eruption column height", "Plinian eruption", "lava tube cave",
        "volcanic glass fracture", "magmatic differentiation", "Vulcanian burst",
        "cinder cone structure", "volcanic island chain", "hotspot migration"
    ],
    "Stellar Evolution": [
        "main sequence star", "red giant expansion", "white dwarf cooling",
        "neutron star rotation", "supernova remnant", "Hertzsprung-Russell diagram",
        "nuclear fusion core", "helium flash ignition", "planetary nebula",
        "Chandrasekhar limit", "core collapse mechanism", "protostellar disk",
        "brown dwarf classification", "pulsating variable", "binary system mass",
        "stellar wind outflow", "magnetar field strength", "accretion luminosity",
        "heavy element nucleosynthesis", "degenerate electron pressure",
        "T Tauri phase", "stellar rotation period", "spectral classification",
        "mass-luminosity relation", "convective envelope", "radiative zone",
        "gravitational contraction", "stellar parallax measurement",
        "Kelvin-Helmholtz timescale", "asymptotic giant branch"
    ],
    "Crystal Structure": [
        "unit cell symmetry", "Bravais lattice type", "Miller index plane",
        "X-ray diffraction pattern", "Bragg reflection angle", "point group",
        "space group notation", "ionic crystal packing", "covalent network solid",
        "metallic bond electron sea", "defect vacancy site", "grain boundary energy",
        "piezoelectric crystal", "anisotropic property", "lattice parameter",
        "phonon dispersion curve", "Brillouin zone boundary", "reciprocal lattice",
        "crystal growth nucleation", "Czochralski method", "epitaxial layer",
        "polymorphic phase transition", "quasicrystal symmetry", "twinning plane",
        "dislocation glide", "stacking fault energy", "Wulff construction",
        "crystal field splitting", "band gap semiconductor", "Fermi surface topology"
    ],
    "Fluid Mechanics": [
        "turbulent flow transition", "laminar boundary layer", "Reynolds number threshold",
        "Navier-Stokes equation", "vortex shedding frequency", "Bernoulli pressure drop",
        "viscous shear stress", "Couette flow profile", "Poiseuille pipe flow",
        "drag coefficient estimation", "lift force generation", "boundary layer separation",
        "wake turbulence mixing", "Stokes creeping flow", "compressible shock wave",
        "Mach number regime", "cavitation bubble", "surface tension capillary",
        "hydraulic jump phenomenon", "open channel flow", "Manning roughness",
        "Froude number criterion", "Euler equation inviscid", "streamline visualization",
        "potential flow theory", "Kelvin-Helmholtz instability", "Rayleigh-Taylor mixing",
        "magnetohydrodynamic force", "non-Newtonian rheology", "microfluidic channel"
    ],
    "Thermodynamic Cycles": [
        "Carnot cycle efficiency", "isothermal expansion", "adiabatic compression",
        "entropy production irreversible", "enthalpy state function", "Gibbs free energy",
        "heat engine work output", "refrigeration coefficient performance",
        "Otto cycle gasoline engine", "Diesel cycle compression ignition",
        "Rankine cycle steam turbine", "Brayton cycle gas turbine",
        "Maxwell thermodynamic relation", "Clausius inequality statement",
        "fugacity activity coefficient", "phase equilibrium coexistence",
        "triple point temperature", "critical point fluid", "van der Waals equation",
        "Joule-Thomson cooling", "exergy destruction analysis", "second law entropy",
        "Boltzmann statistical mechanics", "partition function calculation",
        "microstate counting entropy", "reversible isothermal process",
        "irreversibility heat transfer", "thermodynamic potential", "Legendre transform",
        "Helmholtz free energy minimum"
    ],
    "Organic Synthesis": [
        "carbon-carbon bond formation", "nucleophilic substitution mechanism",
        "electrophilic addition reaction", "Grignard reagent preparation",
        "Wittig olefination product", "Diels-Alder cycloaddition",
        "cross-coupling palladium catalyst", "functional group protection",
        "stereochemistry chiral center", "enantioselective synthesis route",
        "retrosynthetic disconnection", "total synthesis strategy",
        "oxidation state change", "reduction aldehyde alcohol",
        "amide bond coupling", "ester hydrolysis saponification",
        "aromatic ring substitution", "radical chain polymerization",
        "click chemistry azide alkyne", "green chemistry solvent-free",
        "flow chemistry microreactor", "asymmetric catalysis ligand",
        "organometallic intermediate", "transition state theory",
        "activation energy barrier", "reaction yield optimization",
        "column chromatography purification", "NMR characterization",
        "mass spectrometry fragmentation", "infrared absorption peak"
    ],
    "Particle Physics": [
        "Higgs boson detection", "quark flavor generation", "lepton charge conservation",
        "Standard Model symmetry", "strong force gluon exchange",
        "weak interaction beta decay", "electromagnetic photon mediator",
        "neutrino oscillation mass", "antimatter annihilation", "hadron collider energy",
        "cross section measurement", "Feynman diagram calculation",
        "renormalization divergence", "gauge boson propagator",
        "spontaneous symmetry breaking", "CP violation asymmetry",
        "dark matter candidate", "supersymmetry prediction",
        "string theory dimension", "quantum gravity unification",
        "proton decay lifetime", "kaon rare decay", "muon anomalous moment",
        "tau lepton discovery", "charm quark charmonium",
        "bottom quark B-meson", "top quark mass measurement",
        "W boson pair production", "Z boson resonance peak",
        "Cabibbo-Kobayashi-Maskawa matrix"
    ],
    "Seismic Analysis": [
        "P-wave arrival time", "S-wave shear motion", "surface wave dispersion",
        "Richter magnitude scale", "moment magnitude calculation",
        "epicenter triangulation", "focal depth determination", "fault rupture propagation",
        "aftershock sequence decay", "foreshock identification",
        "seismograph recording trace", "strong motion accelerogram",
        "site amplification factor", "liquefaction susceptibility",
        "tsunami generation mechanism", "seismic hazard assessment",
        "return period estimation", "attenuation distance relation",
        "response spectrum analysis", "base isolation design",
        "structural resonance frequency", "soil-structure interaction",
        "active fault mapping", "paleoseismology trench", "geodetic strain rate",
        "InSAR deformation map", "Coulomb stress transfer",
        "earthquake early warning", "induced seismicity injection",
        "moment tensor inversion"
    ],
    "Atmospheric Science": [
        "tropospheric circulation cell", "stratospheric ozone chemistry",
        "mesospheric temperature inversion", "Coriolis force deflection",
        "jet stream meandering", "convective instability mechanism",
        "adiabatic lapse rate", "dew point saturation", "cloud condensation nuclei",
        "precipitation formation process", "cyclone pressure gradient",
        "anticyclone subsidence clearing", "frontal boundary lifting",
        "tropical cyclone intensification", "monsoon seasonal reversal",
        "El Niño ocean coupling", "radiative forcing imbalance",
        "aerosol indirect effect", "greenhouse warming feedback",
        "polar vortex disruption", "sudden stratospheric warming",
        "Hadley cell expansion", "Walker circulation weakening",
        "atmospheric boundary layer", "turbulent heat flux",
        "wind shear thunderstorm", "radar reflectivity rainfall",
        "satellite remote sensing", "weather model ensemble",
        "climate projection scenario"
    ],

    # BIOLOGICAL SCIENCES (10)
    "Enzyme Kinetics": [
        "Michaelis-Menten saturation", "Vmax maximum velocity",
        "Km substrate affinity", "Lineweaver-Burk reciprocal plot",
        "competitive inhibitor binding", "allosteric regulation site",
        "turnover number kcat", "catalytic efficiency ratio",
        "cofactor metal ion requirement", "coenzyme NAD reduction",
        "active site residue", "induced fit conformational change",
        "transition state stabilization", "activation energy lowering",
        "product inhibition feedback", "cooperative binding Hill coefficient",
        "sigmoidal kinetics curve", "ping-pong bi-bi mechanism",
        "ordered sequential binding", "random substrate addition",
        "enzyme immobilization support", "temperature denaturation",
        "pH optimum activity", "irreversible inhibitor covalent",
        "suicide substrate inactivation", "isozyme tissue distribution",
        "regulatory enzyme cascade", "zymogen activation cleavage",
        "multienzyme complex channeling", "kinetic isotope effect"
    ],
    "Coral Reef Ecology": [
        "zooxanthellae photosynthetic symbiosis", "calcium carbonate skeleton",
        "reef framework accretion", "bleaching thermal stress response",
        "crown-of-thorns predation", "herbivore grazer fish",
        "algal turf overgrowth", "nutrient eutrophication impact",
        "ocean acidification dissolution", "larval dispersal connectivity",
        "recruitment settlement cue", "patch reef fragmentation",
        "biodiversity hotspot richness", "trophic cascade effect",
        "keystone predator control", "cleaning station mutualism",
        "cryptic organism diversity", "bioerosion parrotfish grazing",
        "reef crest wave energy", "back reef lagoon sediment",
        "fore reef slope current", "deep mesophotic community",
        "marine protected area effectiveness", "restoration coral gardening",
        "disease white syndrome spread", "spawning synchrony moonlight",
        "symbiont shuffling adaptation", "resilience recovery trajectory",
        "connectivity genetic flow", "microbiome holobiont health"
    ],
    "Neurotransmitter Signaling": [
        "synaptic vesicle release", "action potential propagation",
        "postsynaptic receptor binding", "dopamine reward pathway",
        "serotonin mood regulation", "acetylcholine muscarinic receptor",
        "GABA inhibitory interneuron", "glutamate excitatory synapse",
        "norepinephrine stress response", "endorphin pain modulation",
        "reuptake transporter protein", "monoamine oxidase degradation",
        "second messenger cAMP cascade", "ion channel gating mechanism",
        "long-term potentiation memory", "synaptic plasticity change",
        "dendritic spine morphology", "axon terminal bouton",
        "neuromuscular junction plate", "autonomic ganglionic transmission",
        "neuropeptide modulatory role", "nitric oxide retrograde signal",
        "calcium influx exocytosis", "quantal release probability",
        "miniature end-plate potential", "presynaptic autoreceptor feedback",
        "desensitization receptor downregulation", "excitotoxicity damage",
        "pharmacological agonist affinity", "antagonist competitive blockade"
    ],
    "Photosynthesis Mechanism": [
        "chloroplast thylakoid membrane", "photosystem II water splitting",
        "electron transport chain", "NADPH reduction power",
        "ATP synthase proton gradient", "Calvin cycle carbon fixation",
        "RuBisCO enzyme oxygenation", "light-harvesting antenna complex",
        "chlorophyll a absorption peak", "carotenoid accessory pigment",
        "stroma carbon reduction", "photorespiration wasteful pathway",
        "C4 anatomy bundle sheath", "CAM nocturnal acid storage",
        "quantum yield efficiency", "photoinhibition damage repair",
        "stomatal conductance regulation", "mesophyll cell chloroplast",
        "plastoquinone electron carrier", "ferredoxin final acceptor",
        "cyclic phosphorylation pathway", "non-cyclic linear flow",
        "oxygen evolving complex manganese", "Z-scheme energy diagram",
        "Emerson enhancement effect", "action spectrum wavelength",
        "fluorescence yield measurement", "carbon isotope discrimination",
        "triose phosphate export shuttle", "starch accumulation granule"
    ],
    "Immune Cell Response": [
        "T-cell receptor antigen recognition", "B-cell antibody secretion",
        "natural killer cytotoxicity", "macrophage phagocytosis engulfment",
        "dendritic cell antigen presentation", "MHC class I pathway",
        "cytokine inflammatory cascade", "complement system activation",
        "immunoglobulin class switching", "clonal expansion proliferation",
        "memory cell long-term protection", "regulatory T-cell suppression",
        "helper CD4 T-cell activation", "cytotoxic CD8 killing",
        "inflammatory neutrophil recruitment", "mast cell degranulation histamine",
        "eosinophil parasite defense", "basophil allergic reaction",
        "toll-like receptor innate sensing", "interferon antiviral state",
        "apoptosis programmed death", "autoimmune self-tolerance failure",
        "thymic selection education", "bone marrow maturation",
        "lymph node germinal center", "spleen blood filter function",
        "mucosal immunity IgA barrier", "vaccine adjuvant enhancement",
        "checkpoint inhibitor immunotherapy", "chimeric antigen receptor engineering"
    ],
    "Genetic Regulation": [
        "promoter transcription initiation", "enhancer element distant activation",
        "repressor protein binding operator", "RNA polymerase holoenzyme",
        "transcription factor zinc finger", "epigenetic methylation silencing",
        "histone acetylation relaxation", "chromatin remodeling complex",
        "microRNA post-transcriptional control", "alternative splicing isoform",
        "polyadenylation signal tail", "ribosome translation initiation",
        "codon anticodon base pairing", "ubiquitin proteasome degradation",
        "signal transduction phosphorylation", "gene regulatory network motif",
        "operon lac induction mechanism", "CRISPR-Cas9 genome editing",
        "transposon mobile element insertion", "imprinting parent-of-origin",
        "X-inactivation dosage compensation", "telomere shortening aging",
        "cell cycle checkpoint control", "apoptosis Bcl-2 family",
        "oncogene activation mutation", "tumor suppressor loss function",
        "stem cell pluripotency factor", "differentiation lineage commitment",
        "feedback loop bistability", "noise stochastic gene expression"
    ],
    "Fungal Biology": [
        "mycelium network hyphal branching", "spore dispersal germination",
        "fruiting body mushroom formation", "saprophytic decomposition cellulose",
        "mycorrhizal root association", "arbuscular nutrient exchange interface",
        "ectomycorrhizal mantle sheath", "pathogenic plant infection strategy",
        "toxin secondary metabolite production", "antibiotic penicillin biosynthesis",
        "fermentation ethanol yeast metabolism", "chitin cell wall composition",
        "septate hypha compartment", "coenocytic multinucleate filament",
        "dikaryotic phase nuclear pair", "basidiospore meiotic product",
        "ascospore ascomycete sac formation", "lichen photobiont partnership",
        "endophyte plant tissue colonization", "bioluminescent light emission",
        "ergot alkaloid neurological effect", "truffle underground fruiting",
        "fairy ring growth pattern", "wood-rot brown white decay",
        "biofilm surface attachment", "quorum sensing chemical signal",
        "dimorphic yeast-hypha switch", "parasexual recombination fusion",
        "conidial asexual reproduction", "mating type compatibility locus"
    ],
    "Marine Mammal Behavior": [
        "echolocation sonar click train", "whale song frequency modulation",
        "pod social structure hierarchy", "breaching surface display",
        "deep diving breath hold", "blubber thermoregulation insulation",
        "baleen filter feeding krill", "tooth whale squid predation",
        "migration seasonal route pattern", "calving nursing maternal care",
        "spy-hopping visual scanning", "fluke identification photography",
        "strandings mass beaching event", "acoustic harassment deterrent",
        "bubble net cooperative feeding", "sleep unihemispheric rest",
        "dolphin spinner aerial leap", "porpoise harbor coastal habitat",
        "seal haul-out resting site", "walrus tusk social display",
        "manatee seagrass grazing", "orca cultural tradition hunting",
        "narwhal tusk sensory organ", "beluga Arctic vocalization",
        "sea lion agility swimming", "elephant seal depth record",
        "monk seal endangered population", "river dolphin freshwater adaptation",
        "cetacean intelligence cognition", "anthropogenic noise ship disturbance"
    ],
    "Plant Disease": [
        "fungal pathogen spore infection", "bacterial wilt vascular blockage",
        "viral mosaic symptom pattern", "nematode root knot galling",
        "phytoplasma yellows stunting", "oomycete late blight epidemic",
        "powdery mildew surface growth", "rust pustule uredinium stage",
        "canker bark lesion necrosis", "damping-off seedling collapse",
        "systemic acquired resistance", "hypersensitive response cell death",
        "pathogenesis-related protein induction", "salicylic acid defense signal",
        "jasmonic acid wound response", "phytoalexin antimicrobial compound",
        "race-specific gene-for-gene model", "quantitative partial resistance",
        "vector insect transmission route", "seed-borne internal contamination",
        "cultural practice rotation control", "biological control antagonist",
        "fungicide chemical protectant", "integrated pest management strategy",
        "quarantine exclusion regulation", "resistant cultivar deployment",
        "virulence effector secretion", "avirulence recognition elicitor",
        "epidemiology disease progress curve", "climate warming outbreak prediction"
    ],
    "Deep Sea Ecosystem": [
        "hydrothermal vent chimney formation", "chemosynthetic bacteria energy source",
        "giant tube worm symbiosis colony", "abyssal plain sediment community",
        "bioluminescent organism counter-illumination", "pressure adaptation barophilic",
        "cold seep methane bubble emission", "whale fall succession community",
        "manganese nodule mineral deposit", "hadal zone trench depth",
        "mesopelagic twilight zone migration", "bathypelagic midnight zone isolation",
        "anglerfish lure predation strategy", "giant squid deep encounter",
        "xenophyophore giant foraminiferan", "brittle star carpet density",
        "sea cucumber deposit feeding", "deep coral cold-water reef",
        "sponge garden filter community", "hydrothermal fluid metal sulfide",
        "black smoker high temperature", "white smoker diffuse flow",
        "riftia pachyptila tube dwelling", "yeti crab hairy chelipod",
        "Pompeii worm heat tolerance", "deep-sea mining environmental concern",
        "submersible exploration technology", "remotely operated vehicle sampling",
        "sediment core paleoclimate record", "dissolved oxygen minimum layer"
    ],

    # SPORTS & ATHLETICS (8)
    "Basketball Tactics": [
        "pick-and-roll screen action", "zone defense rotation shift",
        "man-to-man denial pressure", "fast break transition opportunity",
        "half-court offensive set play", "three-point spacing floor",
        "post-up back-to-basket move", "drive-and-kick perimeter pass",
        "high pick-and-pop shooting", "backdoor cut basket finish",
        "inbound sideline special play", "press break advance advance",
        "full-court trapping pressure", "switching defensive assignment",
        "help-side weakside positioning", "box-out rebounding technique",
        "shot clock management pace", "timeout strategy momentum shift",
        "foul trouble rotation adjustment", "double-team ball trap corner",
        "outlet pass fast transition", "clearout isolation one-on-one",
        "horn set wing entry angle", "motion offense read react",
        "dribble hand-off wing action", "flare screen three-pointer",
        "lob alley-oop finish rim", "offensive rebounding putback tip",
        "defensive rebounding box-out seal", "end-game clock management"
    ],
    "Marathon Physiology": [
        "glycogen depletion wall bonking", "aerobic threshold pace zone",
        "lactate accumulation fatigue onset", "VO2max oxygen uptake capacity",
        "running economy stride efficiency", "negative split pacing strategy",
        "carbohydrate loading preparation", "electrolyte sodium replacement",
        "core temperature heat dissipation", "sweat rate hydration balance",
        "muscle fiber type recruitment", "mitochondrial density adaptation",
        "capillary bed perfusion delivery", "fat oxidation energy substrate",
        "central governor fatigue theory", "biomechanical heel toe contact",
        "cadence turnover steps minute", "ground contact braking force",
        "hill gradient power demand", "altitude hypoxic training effect",
        "periodization training block structure", "taper recovery peak performance",
        "overtraining syndrome marker cortisol", "heart rate variability readiness",
        "GPS pace tracking wrist device", "race nutrition gel timing",
        "blister prevention friction management", "compression garment recovery claim",
        "mental toughness determination persistence", "finish sprint kick adrenaline"
    ],
    "Cricket Strategy": [
        "swing bowling seam position", "spin delivery finger wrist",
        "batting stance guard middle", "drive cover straight shot",
        "pull hook boundary stroke", "fielding slip cordon position",
        "wicketkeeper stumping chance", "pace bouncer short delivery",
        "yorker toe-crusher length", "slower ball deception change-up",
        "powerplay field restriction over", "death bowling wide yorker",
        "partnership building rotation strike", "declaration timing test match",
        "follow-on enforcing lead margin", "DRS review decision challenge",
        "run rate required chasing", "net run rate qualification",
        "captain field setting change", "new ball reverse swing condition",
        "pitch deterioration day five", "overhead cloud bowling condition",
        "batting collapse middle order", "tailender nightwatchman role",
        "opener facing new ball seam", "middle order anchoring innings",
        "finisher slog over boundary", "test match draw blocking",
        "limited overs acceleration phase", "all-rounder balance contribution"
    ],
    "Olympic Gymnastics": [
        "vault Yurchenko entry technique", "uneven bars release move catch",
        "balance beam acrobatic series", "floor exercise tumbling pass",
        "pommel horse circle leg swing", "still rings iron cross hold",
        "parallel bars swing release", "high bar giant swing dismount",
        "difficulty score element value", "execution deduction penalty",
        "artistry presentation component", "landing stuck deduction range",
        "routine composition requirement", "start value difficulty tariff",
        "neutral deduction time boundary", "apparatus rotation competition order",
        "qualifying round team final", "all-around combined apparatus",
        "event final top eight", "team score counting best",
        "code of points evaluation", "coaching signal allowed assist",
        "warm-up rotation practice touch", "inquiry score challenge protest",
        "chalk grip preparation surface", "spring-loaded apparatus flexibility",
        "injury rehabilitation comeback", "peak performance age window",
        "mental block fear skill", "choreography musicality expression"
    ],
    "Sailing Navigation": [
        "tacking windward upwind angle", "jibing downwind turn maneuver",
        "spinnaker deployment running reach", "genoa furling headsail trim",
        "mainsail reefing wind reduction", "keel centerboard lateral resistance",
        "rudder helm balance steering", "apparent wind angle indicator",
        "true wind speed direction", "current set drift compensation",
        "tide table timing height", "chart navigation waypoint course",
        "GPS electronic position fix", "radar collision avoidance",
        "VHF radio distress communication", "buoy mark rounding rule",
        "right-of-way starboard tack", "protest flag rule infraction",
        "start line timing approach", "finish line crossing clearance",
        "handicap rating time allowance", "crew weight rail hike",
        "trimming telltale flow indicator", "boom vang tension control",
        "backstay mast bend tuning", "halyard tension luff shape",
        "lee helm weather helm balance", "heeling angle stability limit",
        "man overboard recovery procedure", "anchor holding ground tackle"
    ],
    "Fencing Technique": [
        "foil blade target torso", "epee whole body valid target",
        "saber cutting edge attack", "lunge explosive forward advance",
        "parry deflection lateral blade", "riposte immediate counter-attack",
        "fleche running attack action", "beat attack blade preparation",
        "disengage circular blade evasion", "feint false attack intention",
        "distance measure footwork control", "en garde ready position stance",
        "advance retreat footwork drill", "balestra jump forward attack",
        "piste strip boundary warning", "priority right-of-way determination",
        "corps-a-corps close distance halt", "remise continuation second intention",
        "counter-attack stop hit timing", "compound attack multiple feints",
        "point-in-line extended threat", "preparation blade engagement",
        "electrical scoring apparatus light", "bout match elimination bracket",
        "pool round-robin seeding score", "direct elimination tableau progression",
        "card penalty warning misconduct", "coaching tactical instruction timeout",
        "referee simultaneous attack decision", "video replay challenge review"
    ],
    "Rock Climbing Technique": [
        "crimp grip finger position", "open hand sloper technique",
        "heel hook rotation body", "toe hook compression problem",
        "dyno dynamic movement launch", "campus board strength training",
        "fingerboard hangboard workout", "route reading beta sequence",
        "crux section hardest move", "grade difficulty rating system",
        "lead climbing quickdraw clip", "top rope belayer anchor",
        "bouldering crash pad spot", "trad gear placement protection",
        "cam spring-loaded device crack", "nut passive wedge placement",
        "anchor equalized redundant solid", "rappel descent rope control",
        "figure-eight knot tie-in", "belay device friction brake",
        "chalk drying grip powder", "shoe rubber friction smear",
        "rest position knee bar", "flagging balance body position",
        "drop knee twist rotation", "gaston outward push grip",
        "undercling upside palm hold", "mantle press top-out move",
        "pumped forearm fatigue lactic", "mental commitment head game"
    ],
    "Competitive Swimming": [
        "freestyle front crawl stroke cycle", "backstroke rotation body roll",
        "breaststroke pull timing kick", "butterfly undulation dolphin kick",
        "individual medley transition order", "flip turn streamline push-off",
        "open turn breaststroke touch", "dive start reaction block",
        "underwater breakout stroke count", "drag reduction body position",
        "hand entry catch phase pull", "kick propulsion ankle flexibility",
        "bilateral breathing pattern timing", "negative split race strategy",
        "pace clock interval training", "threshold set lactate management",
        "taper rest competition peak", "shave down resistance reduction",
        "tech suit compression buoyancy", "lane line turbulence reduction",
        "split time lap comparison", "touch pad electronic timing",
        "relay exchange take-over zone", "medley relay backstroke start",
        "distance per stroke efficiency", "stroke rate tempo trainer",
        "sculling drill feel water", "hypoxic set breath control",
        "dryland weight resistance training", "video analysis technique correction"
    ],

    # ARTS & CULTURE (6)
    "Baroque Music": [
        "basso continuo figured bass", "counterpoint fugue voice entry",
        "terraced dynamics loud soft contrast", "ornamentation trill mordent",
        "concerto grosso tutti ripieno", "ritornello form orchestral theme",
        "aria da capo ABA structure", "recitative text setting speech",
        "chorale harmonization four-part voice", "suite dance movement collection",
        "passacaglia ground bass variation", "harpsichord keyboard temperament tuning",
        "violin sonata solo accompaniment", "opera seria heroic subject",
        "cantata sacred text vocal", "overture French dotted rhythm",
        "polyphony independent melodic lines", "modulation key change transition",
        "sequence melodic pattern repetition", "suspension dissonance resolution preparation",
        "imitation canonical following entry", "invertible counterpoint exchange voices",
        "cadence authentic perfect close", "chromaticism expressive alteration",
        "affect doctrine emotional representation", "thorough-bass realization improvisation",
        "trio sonata two melody one bass", "da capo instruction repeat sign",
        "stile antico old style homage", "continuo cello bassoon support"
    ],
    "Ceramic Glazing": [
        "kiln firing temperature cone", "reduction atmosphere oxygen starved",
        "oxidation clear oxygen abundant", "glaze chemistry flux silica alumina",
        "crawling defect surface tension", "pinholing gas bubble escape",
        "crazing network crack thermal", "shivering compression glaze peel",
        "dipping application technique uniform", "spraying atomized glaze coat",
        "brushwork stroke direction layer", "wax resist decoration pattern",
        "underglazing colorant painting bisque", "overglaze enamel low-fire detail",
        "ash glaze wood-fire natural", "celadon jade green iron",
        "tenmoku iron-saturated brown black", "shino carbon-trap orange white",
        "raku rapid cooling crackle", "salt vapor surface reaction",
        "crystalline zinc crystal growth", "lustre metallic sheen coating",
        "terra sigillata burnished slip", "engobe colored clay covering",
        "sgraffito scratching through layer", "majolica tin-opacified painting",
        "porcelain translucency vitrification", "stoneware durability high-fire",
        "earthenware porous low-fire body", "thermal expansion fit match"
    ],
    "Film Cinematography": [
        "depth-of-field bokeh aperture control", "tracking shot dolly movement",
        "crane aerial sweeping overhead", "steadicam fluid walking shot",
        "handheld shaky urgent documentary", "dutch angle tilted unsettling",
        "close-up face emotion detail", "wide establishing location geography",
        "medium shot conversation framing", "over-the-shoulder reverse dialogue",
        "rack focus shift attention depth", "pull focus reveal background",
        "natural lighting window ambient", "three-point key fill back",
        "high-key bright even exposure", "low-key shadow contrast mood",
        "golden hour warm exterior glow", "practical lamp source motivation",
        "color temperature white balance", "gel filter mood coloring",
        "slow motion overcranked frame rate", "time-lapse compressed duration",
        "long take unbroken continuous", "montage editing rhythm compression",
        "aspect ratio widescreen anamorphic", "lens flare optical artifact",
        "silhouette backlit rim edge", "reflection mirror visual metaphor",
        "chiaroscuro light dark painting", "negative space composition emptiness"
    ],
    "Street Art Culture": [
        "spray can aerosol technique pressure", "stencil cut template repeated",
        "wheat paste poster adhesion overnight", "throw-up bubble letter quick",
        "wildstyle interlocking complex lettering", "tag signature identity mark",
        "piece masterpiece elaborate color", "mural large-scale community wall",
        "political protest statement visual", "permission legal commissioned wall",
        "abandoned building gallery urban", "train yard freight rolling stock",
        "cap nozzle width line variation", "fading gradient blending smooth",
        "character figure cartoon illustration", "background landscape setting depth",
        "3D illusion shadow perspective trick", "collective crew collaboration group",
        "documentary photography preservation record", "gentrification displacement controversy",
        "auction gallery commercial market value", "anonymity identity hidden pseudonym",
        "buff removal clean authority erase", "territory marking neighborhood boundary",
        "style evolution influence generation", "sketch blackbook planning design",
        "paint drip run texture effect", "roller large fill coverage area",
        "installation mixed-media sculptural site", "projection light temporary display"
    ],
    "Classical Ballet": [
        "plié bending knee preparation", "relevé rising half-toe full-pointe",
        "tendu stretching foot floor brush", "dégagé extending foot off floor",
        "rond de jambe circular leg", "fondu melting single-leg bend",
        "développé unfolding extended hold", "grand battement high kick throw",
        "pirouette turning spinning balance", "fouetté whipping turning sequence",
        "grand jeté split leap airborne", "assemblé jumping feet together landing",
        "changement changing feet jump", "entrechat beating feet jump crossing",
        "pas de deux partnering supported", "adagio slow sustained movement",
        "allegro fast jumping petite grande", "port de bras arm carriage",
        "épaulement shoulder head direction", "croisé crossed body direction",
        "effacé open body direction stage", "arabesque extended leg behind",
        "attitude lifted bent leg behind", "penché tilting forward extension",
        "balancé rocking swaying step", "bourrée quick small steps traveling",
        "coupé cutting replacing step preparation", "tombé falling weight transfer",
        "glissade gliding connecting step", "sissonne jumping scissor split"
    ],
    "Documentary Filmmaking": [
        "observational cinema fly-on-wall", "participatory filmmaker-subject interaction",
        "expository voice-of-God narration", "poetic associative visual rhythm",
        "reflexive self-aware process revelation", "performative emotional personal essay",
        "interview talking head framing", "B-roll cutaway visual support",
        "archival footage historical evidence", "vérité spontaneous unscripted moment",
        "sound design ambient atmosphere", "score emotional underscore mood",
        "release form consent signature", "ethics representation power dynamics",
        "access trust building relationship", "funding grant proposal submission",
        "festival circuit premiere screening", "distribution platform streaming broadcast",
        "narrative arc structure storytelling", "character development journey transformation",
        "conflict tension dramatic question", "resolution conclusion reflection",
        "pacing editing rhythm duration", "exposition information delivery scene",
        "montage compression time passage", "sequence shot long-take immersion",
        "color grading mood consistency", "graphics animation data visualization",
        "fact-checking verification accuracy", "impact audience social change advocacy"
    ],

    # FOOD & AGRICULTURE (8)
    "Sourdough Fermentation": [
        "wild yeast natural levain culture", "lactobacillus acid bacteria flavor",
        "hydration ratio flour water percentage", "autolyse rest gluten development",
        "bulk fermentation rise volume increase", "stretch-fold strength building technique",
        "shaping tension surface skin formation", "proofing final rise readiness test",
        "scoring slash blade ear opening", "oven spring rapid initial rise",
        "crumb structure open hole airy", "crust caramelization Maillard browning",
        "starter feeding discard refresh cycle", "temperature fermentation speed control",
        "flour protein gluten strength content", "whole grain bran nutrition fiber",
        "retarding cold fridge overnight develop", "float test readiness buoyancy",
        "banneton basket proofing shape support", "Dutch oven steam crust development",
        "preferment poolish biga sponge", "dough extensibility windowpane test",
        "amylase enzyme starch sugar conversion", "acetic acid vinegar tang flavor",
        "crackle crust sound freshness indicator", "scoring pattern decorative design",
        "baking stone thermal mass retention", "steam injection commercial oven",
        "cooling rack structure setting rest", "slicing bread knife serrated tool"
    ],
    "Fermented Beverage": [
        "primary fermentation yeast sugar conversion", "secondary conditioning carbonation clarity",
        "specific gravity hydrometer measurement", "attenuation percentage sugar consumed",
        "hop addition bittering aroma flavor", "malt barley kilning roasting color",
        "mash temperature enzyme rest saccharification", "sparging water grain rinsing extract",
        "boil kettle sterilization hop isomerization", "wort chilling rapid cooling pitch",
        "yeast strain selection ester phenol", "fermentation temperature control profile",
        "dry hopping aroma post-fermentation addition", "bottle conditioning natural carbonation",
        "keg force carbonation pressure serving", "cold crash clarification settling",
        "fining agent gelatin bentonite clarity", "filtration micron particle removal",
        "off-flavor diacetyl butter clean-up", "oxidation stale cardboard prevention",
        "pH acidity mash adjustment water", "water chemistry mineral ion profile",
        "adjunct rice corn sugar lightening", "barrel aging wood oak character",
        "blending batch consistency balance", "wild fermentation spontaneous inoculation",
        "brett brettanomyces funk character", "sour lactobacillus pediococcus kettle",
        "fruit addition secondary flavor infusion", "packaging shelf-life freshness dating"
    ],
    "Cheese Microbiology": [
        "lactic acid bacteria starter culture", "rennet coagulation curd formation",
        "curd cutting whey separation drainage", "salt addition moisture preservation",
        "affinage cave aging maturation time", "rind development surface mold flora",
        "Penicillium camemberti white bloomy rind", "Penicillium roqueforti blue vein internal",
        "Brevibacterium linens washed orange smear", "Propionibacterium eye formation Swiss hole",
        "proteolysis protein breakdown flavor texture", "lipolysis fat breakdown fatty acid",
        "ammoniation deamination pungent ammonia", "crystallization tyrosine calcium lactate",
        "paste texture soft semi-hard firm", "moisture content classification fresh aged",
        "milk source cow sheep goat buffalo", "pasteurization raw milk safety risk",
        "thermophilic mesophilic temperature range starter", "acidification pH drop curd development",
        "stretching pasta filata mozzarella technique", "pressing weight consolidation density",
        "waxing coating oxygen barrier protection", "cloth-bound bandaged traditional aging",
        "cave humidity temperature control environment", "turning flipping wheel rotation care",
        "grading quality inspection evaluation", "terroir origin local character expression",
        "artisan small-batch handmade craft production", "flavor wheel aroma taste vocabulary"
    ],
    "Tropical Fruit Cultivation": [
        "mango grafting rootstock scion variety", "avocado Hass flowering type alternation",
        "pineapple ratoon sucker propagation planting", "banana bunch finger hand harvest",
        "papaya ringspot virus resistant variety", "passion fruit vine trellis pollination",
        "guava psyllid pest management control", "lychee panicle fruit set climate",
        "dragon fruit pitaya night flowering", "jackfruit cauliflorous trunk bearing",
        "durian odor compound sulfur ripeness", "mangosteen slow growth juvenile period",
        "rambutan hairy lychee relative tropical", "starfruit carambola waxy ridged shape",
        "coconut palm copra oil water", "breadfruit starchy carbohydrate staple food",
        "tamarind pod sour pulp seed", "pomelo citrus thick rind bitter",
        "soursop custard apple aromatic flesh", "cherimoya subtropical highland cool",
        "sapodilla chicle gum latex fruit", "longan brown shell translucent flesh",
        "persimmon astringent tannin ripening removal", "kumquat small citrus edible peel",
        "feijoa pineapple guava subtropical fruit", "jaboticaba bark grape cluster",
        "acai berry palm Amazonian harvest", "cacao pod fermentation bean drying",
        "vanilla orchid pollination curing bean", "cardamom rhizome spice capsule seed"
    ],
    "Mushroom Cultivation": [
        "spawn inoculation substrate colonization", "fruiting body pinning primordia formation",
        "humidity chamber misting fog system", "temperature cycling trigger fruiting initiation",
        "substrate preparation pasteurization sterilization", "grain spawn expansion bulk transfer",
        "hardwood sawdust supplement bran mix", "straw chopping soaking inoculation bed",
        "log inoculation plug spawn drill", "indoor controlled environment growth room",
        "oyster pleurotus fast colonizer aggressive", "shiitake lentinula oak log cultivation",
        "lion's mane hericium cascading teeth", "king trumpet eryngii thick stem",
        "reishi ganoderma medicinal polysaccharide extract", "maitake grifola hen-of-woods cluster",
        "enoki flammulina cold temperature elongation", "cordyceps caterpillar fungus supplement",
        "chanterelle wild harvest ectomycorrhizal uncultivated", "morel fire-associated spring flush",
        "contamination trichoderma green mold competitor", "agar plate culture isolation transfer",
        "liquid culture syringe injection inoculation", "monotub shotgun fruiting chamber design",
        "casing layer peat vermiculite moisture", "harvest timing veil break spore drop",
        "drying dehydrator preservation storage method", "fresh shelf-life refrigeration packaging",
        "nutritional value protein fiber content", "medicinal compound beta-glucan extraction"
    ],

    # TECHNOLOGY & ENGINEERING (6)
    "Semiconductor Process": [
        "photolithography mask pattern exposure", "chemical vapor deposition thin film",
        "plasma etching reactive ion removal", "ion implantation doping concentration",
        "thermal oxidation silicon dioxide growth", "metal sputtering interconnect wiring",
        "chemical mechanical planarization polishing", "epitaxial growth single crystal layer",
        "wafer cleaning contamination particle removal", "resist coating spinning photosensitive",
        "alignment overlay registration accuracy", "defect inspection yield improvement",
        "critical dimension measurement linewidth", "etch selectivity material contrast",
        "step coverage conformality topography", "void formation electromigration reliability",
        "copper damascene dual inlaid process", "high-k gate dielectric capacitance",
        "FinFET three-dimensional transistor structure", "EUV extreme ultraviolet wavelength source",
        "multiple patterning pitch splitting resolution", "atomic layer deposition angstrom control",
        "contamination metallic organic ionic particle", "metrology measurement inline monitoring",
        "test structure parametric electrical probe", "reliability stress migration lifetime",
        "package assembly wire bond flip", "die singulation scribing cutting",
        "burn-in accelerated aging screening", "foundry process design kit library"
    ],
    "Rocket Engine Design": [
        "combustion chamber pressure temperature flame", "turbopump oxidizer fuel delivery impeller",
        "nozzle expansion ratio throat exit", "regenerative cooling channel jacket",
        "injector element spray atomization mixing", "propellant grain solid fuel burn",
        "liquid oxygen kerosene hypergolic storable", "specific impulse performance efficiency",
        "thrust-to-weight ratio structural mass", "gimbal steering deflection nozzle pivot",
        "staging separation ignition sequence", "ullage motor settling propellant",
        "cavitation suppression inducer inlet pressure", "material ablative heat shield throat",
        "Inconel superalloy high-temperature turbine blade", "additive manufacturing complex channel print",
        "hot-fire static test acceptance stand", "chamber instability combustion oscillation",
        "acoustic mode resonance damping baffle", "oxidizer-fuel ratio mixture optimal",
        "preburner gas generator drive cycle", "expander cycle closed heated exhaust",
        "full-flow staged dual preburner power", "electric pump motor battery mass",
        "throttle range deep minimum shutdown", "restart multiple ignition reliable relighter",
        "vacuum adaptation altitude compensation bell", "aerospike plug external expansion design",
        "methane hydrogen performance density trade", "reusability landing legs grid fins"
    ],
    "Fiber Optic Communication": [
        "single-mode core diameter wavelength propagation", "multi-mode step graded index fiber",
        "attenuation loss decibel per kilometer", "dispersion pulse broadening bit rate",
        "wavelength division multiplexing channel capacity", "dense WDM narrow spacing grid",
        "erbium-doped fiber amplifier optical gain", "Raman amplification distributed pump",
        "modulation amplitude phase coherent format", "photodetector PIN avalanche responsivity",
        "laser diode DFB direct external modulate", "connector splice loss insertion return",
        "bend radius minimum curvature protection", "cable ribbon loose tube armored",
        "fusion splicing alignment arc heating", "OTDR reflectometry fault distance trace",
        "chromatic dispersion compensation grating", "polarization mode rotation birefringence",
        "nonlinear effect Kerr self-phase modulation", "four-wave mixing crosstalk penalty",
        "forward error correction coding overhead", "coherent detection digital signal process",
        "submarine cable repeater span depth", "datacenter short-reach transceiver pluggable",
        "passive optical network splitting distribution", "free-space optical wireless line-of-sight",
        "photonic integrated circuit chip waveguide", "silicon photonics CMOS compatible platform",
        "quantum key distribution secure channel", "space division multiplexing multicore fiber"
    ],
    "Wind Energy Engineering": [
        "blade aerodynamic profile airfoil design", "tip speed ratio optimal extraction",
        "Betz limit theoretical maximum capture", "rotor diameter swept area power",
        "nacelle housing generator gearbox yaw", "tower foundation monopile gravity base",
        "offshore floating semi-submersible tension moored", "pitch control variable angle feather",
        "yaw alignment wind direction tracking", "power curve speed output relationship",
        "cut-in rated cut-out wind speed", "capacity factor annual energy yield",
        "wake effect downstream turbine loss", "array layout spacing optimization",
        "grid connection transmission cable export", "converter power electronics frequency variable",
        "generator direct-drive permanent magnet doubly-fed", "gearbox planetary epicyclic ratio speed",
        "composite blade manufacturing resin infusion", "fatigue loading cycle stress lifetime",
        "condition monitoring vibration temperature sensing", "preventive maintenance scheduled inspection",
        "lightning protection receptor down conductor", "icing detection de-icing heating blade",
        "noise emission aeroacoustic trailing edge", "bird bat collision mitigation curtailment",
        "environmental impact assessment permitting regulation", "levelized cost energy LCOE economics",
        "repowering decommissioning end-of-life recycling", "hybrid storage battery hydrogen integration"
    ],
    "Autonomous Vehicle": [
        "lidar point cloud 3D object detection", "camera image segmentation lane recognition",
        "radar velocity measurement all-weather tracking", "sensor fusion kalman filter integration",
        "HD map localization lane-level precision", "GNSS GPS positioning correction signal",
        "inertial measurement unit dead reckoning", "simultaneous localization mapping SLAM",
        "path planning trajectory optimization control", "model predictive control steering throttle",
        "behavior planning decision making intention", "traffic sign recognition classification speed",
        "pedestrian detection crossing prediction behavior", "vehicle-to-vehicle communication cooperative awareness",
        "edge case corner case scenario testing", "simulation virtual environment validation mile",
        "safety redundancy fail-operational architecture", "functional safety ASIL level requirement",
        "over-the-air update software deployment", "cybersecurity intrusion detection protection",
        "regulatory framework testing certification approval", "disengagement report intervention metric",
        "occupant monitoring attention drowsiness state", "weather adaptation rain fog snow handling",
        "construction zone navigation cone barrel", "emergency vehicle siren detection yielding",
        "parking automated valet summon feature", "highway on-ramp merge speed matching",
        "intersection negotiation unprotected left turn", "ethical dilemma trolley problem programming"
    ],
    "3D Printing Technology": [
        "fused deposition modeling filament extrusion", "stereolithography UV resin curing laser",
        "selective laser sintering powder bed fusion", "binder jetting adhesive inkjet powder",
        "direct metal laser melting titanium steel", "electron beam melting vacuum chamber",
        "material jetting photopolymer droplet deposition", "layer adhesion interlayer bond strength",
        "support structure overhang removal breakaway", "infill pattern density strength weight",
        "nozzle diameter resolution speed tradeoff", "bed temperature adhesion warping prevention",
        "retraction stringing ooze travel move", "bridging unsupported horizontal span cooling",
        "slicing software toolpath generation Gcode", "build orientation part quality surface",
        "post-processing sanding painting vapor smoothing", "tolerance dimensional accuracy measurement",
        "multi-material dual extrusion color mixing", "continuous fiber reinforcement composite print",
        "bioprinting scaffold tissue cell-laden hydrogel", "concrete construction large-scale building",
        "food printing chocolate paste extrusion", "jewelry casting wax burnout investment",
        "dental crown implant custom digital scan", "aerospace lightweight lattice topology optimized",
        "rapid prototyping iteration design validation", "production batch custom mass customization",
        "quality control defect detection in-situ", "sustainability recycled material closed loop"
    ],

    # HISTORY & GEOGRAPHY (7)
    "Viking Exploration": [
        "longship clinker-built oak construction sailing", "Norse saga oral tradition Icelandic",
        "rune stone inscription memorial carved", "settlement Vinland Norse North America",
        "raid monastery coastal village plunder", "navigation sun stone latitude sailing",
        "Thing assembly law dispute resolution", "jarl chieftain warrior band retinue",
        "Danelaw territory England Anglo-Saxon", "Varangian Guard Byzantine Constantinople service",
        "trade route silver fur amber", "burial ship grave goods afterlife",
        "skald poetry praise commemoration verse", "berserker frenzied warrior fearless battle",
        "shield wall formation defensive tactic", "Odin Thor Freya Norse mythology",
        "Ragnarok end-times prophecy renewal cycle", "dragon prow figurehead intimidation symbol",
        "comb antler craftsmanship personal hygiene", "jewelry brooch arm-ring status display",
        "York Jorvik urban craft market", "Dublin Waterford Irish Viking settlement",
        "Greenland colony Erik climate change", "L'Anse aux Meadows archaeological site",
        "clinker hull overlapping plank construction", "knarr cargo vessel trade transport",
        "navigation bird release coastal landmark", "winter camp overwintering preparation supply",
        "conversion Christianity gradual adoption cross", "runestone memorial death commemoration voyage"
    ],
    "Medieval Castle": [
        "motte bailey wooden tower palisade", "stone keep tower rectangular square",
        "curtain wall enclosure defensive perimeter", "gatehouse portcullis drawbridge entrance defense",
        "arrow loop slit narrow opening fire", "battlement crenel merlon parapet walk",
        "moat water dry ditch obstacle", "barbican outer fortification approach defense",
        "murder hole machicolation overhead dropping", "spiral staircase clockwise defender advantage",
        "great hall feast gathering audience throne", "solar private chamber family apartment",
        "chapel worship spiritual castle community", "dungeon prison below ground confinement",
        "garderobe toilet waste chute exterior", "well water cistern supply siege",
        "buttress support wall structural strength", "vault ceiling stone arch rib",
        "tower corner flanking enfilade fire", "postern small rear secret exit",
        "siege mining undermine collapse tunnel", "trebuchet catapult bombardment stone throw",
        "concentric ring multiple layer defense depth", "earthwork landscape terrain advantage position",
        "garrison troops soldiers manning defense", "provisions storage stockpile siege endurance",
        "construction master mason builder stone", "quarry limestone local material transport",
        "heraldry banner coat-of-arms identity display", "tournament jousting practice combat training"
    ],
    "Silk Road Commerce": [
        "caravan camel train merchant route overland", "oasis town water market resupply stop",
        "silk textile luxury fabric Chinese export", "spice pepper cinnamon cardamom trade value",
        "jade nephrite precious stone craft carving", "paper invention knowledge transmission culture",
        "gunpowder formula military technology spread", "compass navigation direction maritime influence",
        "Sogdian intermediary merchant polyglot network", "Parthian middleman markup price control",
        "Roman gold coin payment demand drain", "Tang dynasty cosmopolitan capital Chang'an",
        "Samarkand Bukhara scholarly city madrasa", "Baghdad Abbasid learning translation house",
        "Kashgar Dunhuang gateway junction crossroads", "Kushan Empire Buddhist art transmission",
        "Mongol Empire safe passage Pax guarantee", "caravanserai inn lodging fortified rest",
        "banditry robbery security escort protection", "tariff tax duty customs revenue",
        "disease plague epidemic transmission spread", "religion Buddhism Islam Christianity diffusion",
        "glass bead Roman production exchange", "lapis lazuli Afghanistan blue luxury",
        "horse Ferghana bloodsweat military breeding", "frankincense myrrh incense Arabia export",
        "coral Mediterranean red luxury jewelry", "cotton textile Indian production technique",
        "porcelain ceramic Chinese kiln masterwork", "slave trade human captive labor transport"
    ],
    "Industrial Revolution": [
        "steam engine Watt condenser efficiency improvement", "textile mill factory spinning weaving machine",
        "coal mining deep shaft ventilation extraction", "iron smelting coke blast furnace production",
        "canal waterway transport barge inland navigation", "railway locomotive passenger freight connection",
        "urbanization city growth slum overcrowding sanitation", "child labor exploitation reform legislation",
        "trade union strike collective bargaining rights", "Luddite machine-breaking resistance displacement",
        "enclosure movement common land displacement rural", "domestic system cottage industry decline",
        "division labor specialization assembly repetition", "standardization interchangeable parts production mass",
        "patent invention intellectual property protection", "capital investment banking finance accumulation",
        "export market colonial raw material source", "population growth demographic transition mortality",
        "public health cholera sanitation sewage reform", "education literacy skill workforce demand",
        "gas lighting street illumination extended day", "telegraph communication rapid long-distance message",
        "photograph documentation record social change", "newspaper press circulation mass information",
        "department store consumer retail shopping urban", "clock time discipline factory schedule shift",
        "pollution smoke river waste environmental damage", "engineering profession institution training civil",
        "social class bourgeoisie proletariat inequality stratification", "empire expansion resource extraction colonial competition"
    ],
    "Ancient Greek Philosophy": [
        "Socratic method questioning dialogue examination", "Platonic form ideal abstract eternal truth",
        "Aristotelian logic syllogism deduction premise conclusion", "pre-Socratic cosmology origin principle arche",
        "Heraclitus flux change fire opposites unity", "Parmenides being unchanging monism rational",
        "Democritus atom void materialism determinism reduction", "Pythagorean number harmony mathematics music",
        "Sophist rhetoric persuasion relative truth skill", "Stoic virtue reason nature fate acceptance",
        "Epicurean pleasure ataraxia atom garden moderate", "Cynic ascetic convention rejection natural simple",
        "Skeptic suspension judgment doubt certainty withhold", "Academy dialogue dialectic political philosopher-king",
        "Lyceum peripatetic observation empirical classification biology", "Republic justice city soul class guardian",
        "Ethics virtue mean happiness eudaimonia character", "Metaphysics substance cause being potentiality",
        "Politics constitution citizen good life community", "Poetics tragedy catharsis mimesis plot character",
        "aporia puzzle difficulty contradiction impasse wonder", "elenchus refutation cross-examination inconsistency",
        "anamnesis recollection knowledge innate soul immortal", "teleology purpose final cause function natural",
        "hylomorphism matter form composite individual change", "cosmological argument first mover unmoved eternal",
        "golden mean moderation excess deficiency virtue", "dialectic thesis antithesis synthesis progression",
        "cave allegory shadow reality enlightenment education", "examined life reflection self-knowledge worth living"
    ],
    "Arctic Tundra Ecology": [
        "permafrost active layer seasonal thaw", "caribou reindeer migration herd calving",
        "lichen slow growth symbiotic pioneer", "polar bear sea ice hunting seal",
        "muskox defensive circle qiviut wool", "Arctic fox winter white coat camouflage",
        "tundra swan nesting wetland breeding", "lemming population cycle boom crash",
        "snow bunting migratory songbird arrival", "snowy owl hunting vole open ground",
        "patterned ground frost heave polygon", "thermokarst thaw lake collapse subsidence",
        "ice wedge contraction crack winter", "pingo mound ice core uplift",
        "treeline ecotone transition boreal boundary", "growing season short photoperiod continuous",
        "alpine tundra elevation mountain above", "cushion plant wind exposure adaptation",
        "moss carpet moisture retention insulation", "wildflower bloom brief summer color",
        "migratory bird staging flyway stopover", "insect emergence brief warm pulse",
        "soil carbon storage organic matter frozen", "methane emission thaw release greenhouse",
        "indigenous Inuit traditional knowledge hunting", "climate warming amplification rapid change",
        "sea ice retreat extent minimum record", "coastal erosion wave permafrost exposure",
        "scientific monitoring station long-term data", "conservation protected area designation management"
    ],
    "Coral Atoll Geography": [
        "lagoon shallow protected calm water", "reef flat exposed low tide platform",
        "pass channel current exchange ocean lagoon", "motu islet sand gravel vegetated",
        "coconut palm dominant vegetation strand", "seabird colony nesting ground guano",
        "sea level rise vulnerability inundation", "freshwater lens aquifer rainfall recharge",
        "subsistence fishing traditional livelihood resource", "pearl farming oyster cultivation industry",
        "cyclone typhoon storm surge damage exposure", "coral rubble storm deposit ridge",
        "beach sand formation erosion accretion", "mangrove establishment shoreline protection nursery",
        "tuna migration pelagic fishing zone", "marine conservation biodiversity sanctuary protection",
        "climate adaptation resilience planning community", "waste management challenge isolated disposal",
        "solar energy renewable power island supply", "desalination freshwater production technology",
        "transport inter-island boat ferry connection", "traditional navigation outrigger canoe voyage",
        "cultural heritage preservation identity language", "population density urbanization capital atoll",
        "tourism diving snorkeling revenue economy", "copra coconut dried export commodity",
        "land tenure communal customary ownership rights", "elevation maximum meters low vulnerability",
        "geological origin volcanic hotspot subsidence", "Darwin theory fringing barrier atoll sequence"
    ],
}

# Verify count
K = len(CATEGORIES)
print(f"Categories defined: {K}")
print(f"Using {K} categories")

# ================================================================
# CONFUSER PAIRS: categories that share some keywords
# These create genuine confusion for the oracle model → heavy gap tail
# ================================================================
CONFUSER_PAIRS = [
    # Science pairs with overlapping concepts
    ("Volcanic Activity", "Seismic Analysis"),  # both about Earth, tremors, etc
    ("Stellar Evolution", "Particle Physics"),   # both physics, nuclear, energy
    ("Fluid Mechanics", "Thermodynamic Cycles"), # both physics, equations, energy
    ("Crystal Structure", "Organic Synthesis"),   # both molecular, structure
    ("Enzyme Kinetics", "Photosynthesis Mechanism"),  # both biochemistry
    ("Neurotransmitter Signaling", "Immune Cell Response"),  # both biology, cells
    ("Genetic Regulation", "Plant Disease"),      # both biology, genes, mechanisms
    ("Coral Reef Ecology", "Deep Sea Ecosystem"),  # both marine biology
    ("Fungal Biology", "Mushroom Cultivation"),    # both fungi
    ("Marine Mammal Behavior", "Coral Reef Ecology"),  # both marine
    # Cross-domain pairs
    ("Sourdough Fermentation", "Fermented Beverage"),  # both fermentation
    ("Cheese Microbiology", "Enzyme Kinetics"),   # both microbiology
    ("Film Cinematography", "Documentary Filmmaking"),  # both film
    ("Semiconductor Process", "Crystal Structure"),  # both materials
    ("Rocket Engine Design", "Fluid Mechanics"),   # both engineering physics
    ("Wind Energy Engineering", "Atmospheric Science"),  # both air/energy
    ("Viking Exploration", "Sailing Navigation"),  # both ships/navigation
    ("Medieval Castle", "Fencing Technique"),       # both medieval combat
    ("Classical Ballet", "Olympic Gymnastics"),     # both body movement
    ("Marathon Physiology", "Competitive Swimming"),  # both athletic physiology
    ("Coral Atoll Geography", "Coral Reef Ecology"),  # both coral
    ("Arctic Tundra Ecology", "Deep Sea Ecosystem"),  # both extreme environments
    ("Industrial Revolution", "Semiconductor Process"),  # both manufacturing
    ("3D Printing Technology", "Semiconductor Process"),  # both fabrication
    ("Autonomous Vehicle", "Fiber Optic Communication"),  # both high-tech
]

# Shared keywords between confuser pairs
SHARED_KEYWORDS = {
    ("Volcanic Activity", "Seismic Analysis"): [
        "tectonic plate boundary", "ground shaking intensity", "geological hazard assessment"
    ],
    ("Stellar Evolution", "Particle Physics"): [
        "nuclear reaction energy", "high-energy collision event", "fundamental force interaction"
    ],
    ("Fluid Mechanics", "Thermodynamic Cycles"): [
        "pressure gradient flow", "energy conservation principle", "engineering system efficiency"
    ],
    ("Crystal Structure", "Organic Synthesis"): [
        "molecular arrangement geometry", "chemical bond analysis", "structural characterization technique"
    ],
    ("Enzyme Kinetics", "Photosynthesis Mechanism"): [
        "biochemical reaction rate", "protein complex function", "energy conversion pathway"
    ],
    ("Neurotransmitter Signaling", "Immune Cell Response"): [
        "cell receptor activation", "signaling cascade pathway", "biological defense mechanism"
    ],
    ("Genetic Regulation", "Plant Disease"): [
        "gene expression control", "resistance mechanism defense", "molecular pathway regulation"
    ],
    ("Coral Reef Ecology", "Deep Sea Ecosystem"): [
        "marine biodiversity conservation", "ocean environment adaptation", "symbiotic organism relationship"
    ],
    ("Fungal Biology", "Mushroom Cultivation"): [
        "fungal growth condition", "substrate colonization process", "spore production dispersal"
    ],
    ("Marine Mammal Behavior", "Coral Reef Ecology"): [
        "marine ecosystem interaction", "ocean habitat conservation", "species population dynamics"
    ],
    ("Sourdough Fermentation", "Fermented Beverage"): [
        "microbial culture activity", "fermentation temperature timing", "flavor development process"
    ],
    ("Cheese Microbiology", "Enzyme Kinetics"): [
        "bacterial metabolic activity", "protein transformation process", "pH environment effect"
    ],
    ("Film Cinematography", "Documentary Filmmaking"): [
        "camera angle composition choice", "visual storytelling technique", "editing post-production craft"
    ],
    ("Semiconductor Process", "Crystal Structure"): [
        "material purity requirement", "surface defect analysis", "thin film growth technique"
    ],
    ("Rocket Engine Design", "Fluid Mechanics"): [
        "high-pressure flow dynamics", "combustion gas expansion", "nozzle flow acceleration"
    ],
    ("Wind Energy Engineering", "Atmospheric Science"): [
        "wind speed measurement profile", "atmospheric boundary condition", "weather pattern influence"
    ],
    ("Viking Exploration", "Sailing Navigation"): [
        "maritime voyage planning", "wind sea condition reading", "coastal navigation technique"
    ],
    ("Medieval Castle", "Fencing Technique"): [
        "combat defensive strategy", "weapon handling skill", "physical confrontation technique"
    ],
    ("Classical Ballet", "Olympic Gymnastics"): [
        "body movement precision", "artistic performance scoring", "physical flexibility training"
    ],
    ("Marathon Physiology", "Competitive Swimming"): [
        "athletic endurance training", "cardiovascular fitness adaptation", "competition race strategy"
    ],
    ("Coral Atoll Geography", "Coral Reef Ecology"): [
        "coral formation growth structure", "marine ecosystem vulnerability", "tropical ocean environment"
    ],
    ("Arctic Tundra Ecology", "Deep Sea Ecosystem"): [
        "extreme environment adaptation", "isolated ecosystem dynamics", "cold environment biology"
    ],
    ("Industrial Revolution", "Semiconductor Process"): [
        "manufacturing production technology", "industrial process innovation", "mass production technique"
    ],
    ("3D Printing Technology", "Semiconductor Process"): [
        "layer-by-layer fabrication method", "precision manufacturing control", "material deposition technique"
    ],
    ("Autonomous Vehicle", "Fiber Optic Communication"): [
        "signal processing algorithm", "high-speed data transmission", "system reliability engineering"
    ],
}


# ================================================================
# Text generation templates (diverse, no category names)
# ================================================================
TEMPLATES = [
    "{kw1} represents a critical aspect of this domain. Combined with {kw2}, it reveals important dynamics related to {kw3}.",
    "Research into {kw1} has established fundamental connections with {kw2}. The implications for {kw3} drive current investigations forward.",
    "Understanding the interplay between {kw1} and {kw2} requires expertise in {kw3}. These relationships define the field's core challenges.",
    "The mechanism underlying {kw1} directly affects {kw2} in measurable ways. When {kw3} is also considered, the complexity increases significantly.",
    "Practitioners focus extensively on {kw1} because of its influence on {kw2}. Progress in {kw3} often depends on mastering both.",
    "Recent advances have connected {kw1} with previously unrelated observations about {kw2}. The role of {kw3} in mediating these connections is now established.",
    "The challenge of {kw1} becomes especially acute when {kw2} conditions prevail. Standard approaches to {kw3} must be adapted accordingly.",
    "Quantitative analysis of {kw1} and {kw2} produces consistent patterns. These patterns correlate with {kw3} measurements across multiple studies.",
    "The fundamental principles governing {kw1} extend naturally to understanding {kw2}. Both depend critically on the behavior of {kw3}.",
    "When {kw1} is disrupted, the effects on {kw2} become immediately apparent. Recovery typically requires intervention at the level of {kw3}.",
    "Training in {kw1} builds essential foundations for working with {kw2}. Advanced applications require integrating {kw3} into the framework.",
    "Historical development of {kw1} parallels progress in understanding {kw2}. Modern approaches now unify both through {kw3}.",
    "The practical application of {kw1} demands precise control of {kw2}. Failure to manage {kw3} leads to suboptimal outcomes.",
    "Experts characterize the relationship between {kw1} and {kw2} as fundamental. The variable {kw3} modulates this relationship in context-dependent ways.",
    "Observation of {kw1} under controlled conditions reveals systematic dependence on {kw2}. The strength of this dependence varies with {kw3}.",
    "Current models of {kw1} incorporate {kw2} as a primary factor. Secondary effects from {kw3} are increasingly recognized as important.",
    "The optimization of {kw1} requires balancing against {kw2} constraints. Best results emerge when {kw3} is simultaneously considered.",
    "Field studies confirm that {kw1} and {kw2} interact through complex feedback loops. The role of {kw3} in these loops was recently clarified.",
    "Standard protocols for {kw1} specify particular attention to {kw2} parameters. Deviations in {kw3} can compromise the entire process.",
    "Theoretical frameworks explaining {kw1} predict specific behaviors of {kw2}. Testing these predictions using {kw3} measurements validates the approach.",
]


def generate_documents(n_per_category=100, confuser_frac=0.5, seed=42):
    """Generate documents with many 'confuser' docs that use ONLY shared/partner keywords.
    
    Confuser types:
    - Type A (20%): Uses ONLY shared keywords (no unique category keywords) → gap ≈ 0
    - Type B (15%): Uses 1 shared + 1 partner keyword → gap near 0
    - Type C (15%): Uses 1 own + 1 shared keyword → small gap
    - Normal (50%): Uses 3 keywords from own category → large gap
    """
    random.seed(seed)
    np.random.seed(seed)

    documents = []
    category_names = list(CATEGORIES.keys())

    # Build reverse lookup: category -> list of confuser partners + shared keywords
    confuser_lookup = {name: [] for name in category_names}
    for (cat_a, cat_b), shared_kws in SHARED_KEYWORDS.items():
        confuser_lookup[cat_a].append((cat_b, shared_kws))
        confuser_lookup[cat_b].append((cat_a, shared_kws))

    for cat_idx, cat_name in enumerate(category_names):
        keywords = CATEGORIES[cat_name]
        confusers = confuser_lookup[cat_name]

        for doc_i in range(n_per_category):
            template = random.choice(TEMPLATES)

            # Determine document type
            r = random.random()
            has_confusers = len(confusers) > 0

            if r < 0.20 and has_confusers:
                # Type A: ONLY shared keywords → nearly zero gap
                partner_name, shared_kws = random.choice(confusers)
                all_kws = list(shared_kws)  # all 3 shared keywords
                is_confuser = True
            elif r < 0.35 and has_confusers:
                # Type B: shared + partner keywords → gap near 0
                partner_name, shared_kws = random.choice(confusers)
                partner_kws = CATEGORIES[partner_name]
                all_kws = random.sample(shared_kws, min(2, len(shared_kws)))
                all_kws.append(random.choice(partner_kws))
                is_confuser = True
            elif r < 0.50 and has_confusers:
                # Type C: 1 own + shared → small gap
                partner_name, shared_kws = random.choice(confusers)
                all_kws = [random.choice(keywords)]
                all_kws.extend(random.sample(shared_kws, min(2, len(shared_kws))))
                is_confuser = True
            else:
                # Normal: 3 keywords from own category → large gap
                all_kws = random.sample(keywords, min(3, len(keywords)))
                is_confuser = False

            # Format template
            kw_dict = {}
            for i in range(3):
                kw_dict[f'kw{i+1}'] = all_kws[i] if i < len(all_kws) else random.choice(keywords)

            text = template.format(**kw_dict)

            documents.append({
                'text': text,
                'category_idx': cat_idx,
                'category_name': cat_name,
                'is_confuser': is_confuser,
            })

    random.shuffle(documents)
    return documents, category_names


def build_bow_features(documents, category_names):
    """Build bag-of-words features and project to lower dimension."""
    # Build vocabulary from all keywords
    all_keywords = []
    keyword_to_idx = {}
    for cat_name, kws in CATEGORIES.items():
        for kw in kws:
            if kw not in keyword_to_idx:
                keyword_to_idx[kw] = len(all_keywords)
                all_keywords.append(kw)

    # Add shared keywords
    for shared_kws in SHARED_KEYWORDS.values():
        for kw in shared_kws:
            if kw not in keyword_to_idx:
                keyword_to_idx[kw] = len(all_keywords)
                all_keywords.append(kw)

    V = len(all_keywords)
    print(f"  Vocabulary size: {V}")

    # Build document BoW features
    N = len(documents)
    doc_bow = np.zeros((N, V))
    for i, doc in enumerate(documents):
        text_lower = doc['text'].lower()
        for kw, idx in keyword_to_idx.items():
            if kw.lower() in text_lower:
                doc_bow[i, idx] = 1.0

    # Build arm BoW features (category keyword indicators)
    K = len(category_names)
    arm_bow = np.zeros((K, V))
    for k, cat_name in enumerate(category_names):
        for kw in CATEGORIES[cat_name]:
            if kw in keyword_to_idx:
                arm_bow[k, keyword_to_idx[kw]] = 1.0

    # Shared keywords get HIGH weight (15.0) in BOTH paired arms
    # This makes confuser pairs produce nearly-equal scores → small gaps (Axis C)
    # while unique keywords (weight 1.0) still allow the oracle to discriminate
    SHARED_WEIGHT = 15.0
    for (cat_a, cat_b), shared_kws in SHARED_KEYWORDS.items():
        idx_a = category_names.index(cat_a)
        idx_b = category_names.index(cat_b)
        for kw in shared_kws:
            if kw in keyword_to_idx:
                arm_bow[idx_a, keyword_to_idx[kw]] = SHARED_WEIGHT
                arm_bow[idx_b, keyword_to_idx[kw]] = SHARED_WEIGHT

    print(f"  Document BoW shape: {doc_bow.shape}")
    print(f"  Arm BoW shape: {arm_bow.shape}")
    print(f"  Avg keywords per doc: {doc_bow.sum(axis=1).mean():.1f}")
    print(f"  Avg keywords per arm: {arm_bow.sum(axis=1).mean():.1f}")

    return doc_bow, arm_bow, all_keywords, keyword_to_idx


def project_features(doc_bow, arm_bow, d=200, seed=42):
    """Project BoW features to d dimensions using random projection."""
    np.random.seed(seed)
    V = doc_bow.shape[1]

    # Random projection matrix (Gaussian)
    W = np.random.randn(V, d) / np.sqrt(d)

    doc_proj = doc_bow @ W
    arm_proj = arm_bow @ W

    # Normalize
    doc_norms = norm(doc_proj, axis=1, keepdims=True)
    doc_norms[doc_norms < 1e-8] = 1.0
    doc_proj_normed = doc_proj / doc_norms

    arm_norms = norm(arm_proj, axis=1, keepdims=True)
    arm_norms[arm_norms < 1e-8] = 1.0
    arm_proj_normed = arm_proj / arm_norms

    return doc_proj_normed, arm_proj_normed


def verify_diagnostics(doc_features, arm_features, ys, T=2000):
    """Check all 3 axes."""
    K = arm_features.shape[0]
    N = doc_features.shape[0]
    d = doc_features.shape[1]

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTICS (T={T}, K={K}, d={d}, N={N})")
    print(f"{'=' * 70}")

    # AXIS B: Coherence
    cos_matrix = arm_features @ arm_features.T
    np.fill_diagonal(cos_matrix, 2.0)
    min_cos = cos_matrix.min(axis=1)
    alpha_hat = np.median(min_cos)
    print(f"\nAxis B (Coherence): α̂ = {alpha_hat:.4f}")
    print(f"  Threshold: < 0.1")
    axis_b = alpha_hat < 0.1
    print(f"  STATUS: {'DEVIATING ✓' if axis_b else 'SATURATED ✗'}")

    # AXIS A: Effective rank
    np.random.seed(42)
    n_sample = min(5000, N)
    sample_idx = np.random.choice(N, n_sample, replace=False)
    sample_arms = np.random.choice(K, n_sample, replace=True)
    phi = doc_features[sample_idx] * arm_features[sample_arms]

    # Remove zero rows (from zero norm docs)
    row_norms = norm(phi, axis=1)
    valid = row_norms > 1e-8
    phi = phi[valid]

    cov = np.cov(phi.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = eigvals.sum() / eigvals.max()
    ratio = r_hat / np.sqrt(T)
    print(f"\nAxis A (Effective Rank): r̂ = {r_hat:.2f}")
    print(f"  r̂/√T = {ratio:.4f}")
    print(f"  Threshold: >= 1")
    axis_a = ratio >= 1
    print(f"  STATUS: {'DEVIATING ✓' if axis_a else 'SATURATED ✗'}")
    print(f"  → Deviates for T ≤ {r_hat**2:.0f}")

    # AXIS C: Gap distribution
    # Use theta*=ones (reward = inner product, simpler and deterministic)
    print(f"\nAxis C (Gap Distribution):")
    theta_oracle = np.ones(d)

    # Compute gaps
    n_test = min(2000, N)
    test_idx = np.random.choice(N, n_test, replace=False)
    gaps = []
    for idx in test_idx:
        phi_all = doc_features[idx:idx+1] * arm_features  # (K, d)
        scores = phi_all @ theta_oracle
        sorted_s = np.sort(scores)[::-1]
        gaps.append(sorted_s[0] - sorted_s[1])

    gaps = np.array(gaps)
    for thresh in [0.001, 0.005, 0.01, 0.05, 0.1]:
        print(f"  frac(gap ≤ {thresh}) = {np.mean(gaps <= thresh):.4f}")
    frac_heavy = np.mean(gaps <= 0.05)
    axis_c = frac_heavy > 0.5
    print(f"  Threshold: frac(gap ≤ 0.05) > 0.5")
    print(f"  STATUS: {'DEVIATING ✓' if axis_c else 'SATURATED ✗'}")
    print(f"  Gap percentiles: p10={np.percentile(gaps,10):.4f}, p25={np.percentile(gaps,25):.4f}, p50={np.percentile(gaps,50):.4f}")

    # Summary
    n_dev = sum([axis_a, axis_b, axis_c])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {n_dev}/3 axes deviate")
    if n_dev >= 3:
        print("→ REGIME: warm-start-essential ★")
    elif n_dev >= 2:
        print("→ REGIME: warm-start-likely")
    else:
        print("→ REGIME: LinUCB-suffices")
    print(f"{'=' * 70}")

    return {'alpha_hat': alpha_hat, 'r_hat': r_hat, 'ratio': ratio,
            'gap_frac': frac_heavy, 'n_deviate': n_dev,
            'axis_a': axis_a, 'axis_b': axis_b, 'axis_c': axis_c}


if __name__ == '__main__':
    print("=" * 70)
    print("SYNTHETIC TEXT DATASET v3")
    print(f"{K} categories, with {len(CONFUSER_PAIRS)} confuser pairs")
    print("=" * 70)

    # Generate documents
    documents, category_names = generate_documents(n_per_category=100, seed=42)
    print(f"\nTotal documents: {len(documents)}")
    n_confuser = sum(1 for d in documents if d['is_confuser'])
    print(f"Confuser documents: {n_confuser} ({100*n_confuser/len(documents):.1f}%)")
    print(f"\nSample texts:")
    for i in range(3):
        print(f"  [{documents[i]['category_name']}{'*' if documents[i]['is_confuser'] else ''}] {documents[i]['text'][:100]}...")

    # Build BoW features
    print("\nBuilding bag-of-words features...")
    doc_bow, arm_bow, all_keywords, kw_to_idx = build_bow_features(documents, category_names)

    # Project and test
    ys = np.array([d['category_idx'] for d in documents])

    d = 150
    print(f"\n{'='*40}")
    print(f"Random projection to d={d}")
    print(f"{'='*40}")
    doc_proj, arm_proj = project_features(doc_bow, arm_bow, d=d)
    diag = verify_diagnostics(doc_proj, arm_proj, ys, T=2000)

    # Save the dataset
    print(f"\n\nSaving dataset with d={d}, T=2000 configuration...")

    dataset = {
        'documents': documents,
        'category_names': category_names,
        'doc_features': doc_proj,
        'arm_features': arm_proj,
        'doc_bow': doc_bow,
        'arm_bow': arm_bow,
        'ys': ys,
        'all_keywords': all_keywords,
        'keyword_to_idx': kw_to_idx,
        'n_categories': K,
        'n_documents': len(documents),
        'confuser_pairs': CONFUSER_PAIRS,
    }

    with gzip.open('synthetic_text_dataset_v3.pkl.gz', 'wb') as f:
        pickle.dump(dataset, f)
    print("Saved to synthetic_text_dataset_v3.pkl.gz")
