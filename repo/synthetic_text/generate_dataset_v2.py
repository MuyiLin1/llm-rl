"""
generate_dataset_v2.py

Revised synthetic text dataset generation that achieves warm-start-essential regime.

Key changes from v1:
- 30+ keywords per category for embedding diversity
- Category name NEVER appears in text
- Paragraph-length documents with varied structure
- Cross-category "confuser" keywords for heavy gap tail
- T=5000 subset for experiments (full 50K also saved)
"""

import numpy as np
import random
import gzip
import pickle
import torch
from collections import defaultdict

# ================================================================
# STEP 1: Define 100 diverse categories with RICH vocabularies
# ================================================================
# Fewer categories (100) but MUCH richer per-category vocabulary
# to boost within-category diversity in embedding space.

CATEGORIES = {
    # PHYSICAL SCIENCES (20)
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

    # BIOLOGICAL SCIENCES (20)
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

    # SPORTS & ATHLETICS (15)
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

    # ARTS & CULTURE (15)
    "Baroque Music Composition": [
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
        "pas de deux partnering supported",  "adagio slow sustained movement",
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

    # FOOD & AGRICULTURE (10)
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
    "Fermented Beverage Production": [
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

    # TECHNOLOGY & ENGINEERING (15)
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
    "Autonomous Vehicle Systems": [
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

    # HISTORY & SOCIETY (5)
    "Viking Age Exploration": [
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
    "Medieval Castle Architecture": [
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
    "Industrial Revolution Impact": [
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
}

assert len(CATEGORIES) == 50, f"Expected 50 categories, got {len(CATEGORIES)}"


# ================================================================
# STEP 2: Diverse text generation
# ================================================================

# Many varied templates - different lengths, structures, and perspectives
TEMPLATES = [
    # Short factual (2 sentences)
    "{kw1} represents a fundamental aspect of this field. When combined with {kw2}, researchers observe significant effects on {kw3}.",
    "Recent studies have focused on {kw1} and its relationship to {kw2}. The implications for {kw3} continue to drive new investigations.",
    "Understanding {kw1} requires careful analysis of {kw2}. Experts agree that {kw3} plays a critical supporting role.",

    # Medium explanatory (3 sentences)
    "The phenomenon of {kw1} has been documented extensively in recent years. Its connection to {kw2} suggests a deeper underlying mechanism. Furthermore, {kw3} provides additional evidence for this theoretical framework.",
    "When examining {kw1}, one must consider the role of {kw2} in shaping outcomes. This interaction produces measurable effects on {kw3}. The combined influence creates complex dynamics worth studying.",
    "Practitioners have long recognized that {kw1} and {kw2} are intimately connected. The bridge between them often involves {kw3}. New techniques now allow precise measurement of these relationships.",

    # Longer narrative (4+ sentences)
    "The history of {kw1} begins with early observations about {kw2}. Over decades, understanding evolved through careful documentation of {kw3}. Today, the field recognizes the interconnected nature of these phenomena. Current work pushes boundaries by exploring how {kw4} relates to established principles.",
    "A comprehensive examination of {kw1} reveals multiple layers of complexity. At the surface level, {kw2} dominates observable behavior. Deeper analysis shows {kw3} operating through subtle mechanisms. The interplay between {kw4} and these factors creates emergent properties not predicted by simple models.",
    "Specialists working with {kw1} frequently encounter challenges related to {kw2}. Standard approaches to {kw3} sometimes prove insufficient in extreme conditions. Novel solutions incorporate {kw4} to overcome these limitations. The resulting improvements demonstrate the value of interdisciplinary thinking.",

    # Problem-solution structure
    "One persistent challenge involves {kw1} interfering with {kw2}. Traditional methods relied on {kw3} to address this issue. However, modern approaches incorporating {kw4} have proven more effective. The key breakthrough came from understanding the underlying mechanism connecting these elements.",

    # Comparison/contrast
    "While {kw1} and {kw2} share certain surface similarities, their fundamental mechanisms differ substantially. The former depends primarily on {kw3}, whereas the latter relies on {kw4}. These differences have practical implications for how professionals approach each phenomenon.",

    # Process description
    "The process begins with careful preparation of {kw1}. Next, {kw2} is introduced under controlled conditions. As {kw3} develops, practitioners monitor for signs of {kw4}. The entire sequence requires precise timing and environmental control.",

    # Observational
    "Field observations consistently show {kw1} occurring in conjunction with {kw2}. The pattern becomes especially pronounced when {kw3} conditions are present. Seasonal variation in {kw4} adds another layer of complexity to these dynamics.",

    # Technical report style
    "Analysis of {kw1} parameters indicates significant correlation with {kw2} measurements. Statistical evaluation of {kw3} data confirms this relationship across multiple independent observations. The effect size suggests {kw4} may serve as a reliable indicator.",

    # Review/survey style
    "Multiple independent groups have reported findings related to {kw1} and {kw2}. The consensus view supports a mechanism involving {kw3}. Remaining controversies center on the role of {kw4} and its magnitude of contribution.",

    # Practical instruction
    "Proper handling of {kw1} requires attention to {kw2} at every stage. Failure to account for {kw3} can lead to suboptimal results. Best practices recommend monitoring {kw4} throughout the entire process.",

    # Question-driven
    "What determines the relationship between {kw1} and {kw2}? Evidence suggests that {kw3} plays a mediating role. When {kw4} conditions change, the dynamics shift in predictable ways.",

    # Cause-effect
    "Changes in {kw1} directly impact the behavior of {kw2}. This cascade effect propagates through {kw3} before ultimately influencing {kw4}. The time scale of these interactions varies depending on environmental conditions.",

    # Classification/taxonomy
    "Experts classify {kw1} into several distinct categories based on {kw2} characteristics. The most common variety involves {kw3} as the primary feature. Less frequent forms exhibit {kw4} instead.",

    # Historical development
    "Early work on {kw1} focused primarily on {kw2}. Subsequent discoveries about {kw3} transformed the field's understanding. Modern perspectives now integrate {kw4} into a unified framework.",
]


def generate_documents_v2(n_per_category=250, seed=42):
    """Generate diverse text documents. No category names in text."""
    random.seed(seed)
    np.random.seed(seed)

    documents = []
    category_names = list(CATEGORIES.keys())

    for cat_idx, cat_name in enumerate(category_names):
        keywords = CATEGORIES[cat_name]
        for doc_i in range(n_per_category):
            # Pick a random template
            template = random.choice(TEMPLATES)

            # Count how many keywords the template needs (kw1-kw4)
            n_kw_needed = template.count('{kw')
            # Sample MORE keywords than needed and use random subsets
            kws = random.sample(keywords, min(n_kw_needed + 2, len(keywords)))

            # Format with available keywords
            fmt_dict = {}
            for i in range(n_kw_needed):
                fmt_dict[f'kw{i+1}'] = kws[i] if i < len(kws) else random.choice(keywords)

            text = template.format(**fmt_dict)

            # Add random additional sentence from other keywords (50% chance)
            if random.random() < 0.5:
                extra_kw = random.choice(keywords)
                extra_templates = [
                    f" Additionally, {extra_kw} provides important context.",
                    f" The significance of {extra_kw} should not be underestimated.",
                    f" Related work on {extra_kw} supports these findings.",
                    f" Observations about {extra_kw} add further evidence.",
                    f" The connection to {extra_kw} remains an active area of study.",
                ]
                text += random.choice(extra_templates)

            documents.append({
                'text': text,
                'category_idx': cat_idx,
                'category_name': cat_name,
            })

    random.shuffle(documents)
    return documents, category_names


def build_dataset_v2(documents, category_names, st_model_name='sentence-transformers/bert-base-nli-mean-tokens'):
    """Embed everything and build the bandit dataset."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading SentenceTransformer: {st_model_name}")
    st_model = SentenceTransformer(st_model_name)

    # Embed category names
    print(f"Embedding {len(category_names)} category names...")
    category_embeddings = st_model.encode(category_names, batch_size=64, show_progress_bar=True)
    category_embeddings = torch.FloatTensor(category_embeddings)
    print(f"  Category embeddings shape: {category_embeddings.shape}")

    # Embed documents
    print(f"Embedding {len(documents)} documents...")
    doc_texts = [d['text'] for d in documents]
    doc_embeddings = st_model.encode(doc_texts, batch_size=256, show_progress_bar=True)
    doc_embeddings = torch.FloatTensor(doc_embeddings)
    print(f"  Document embeddings shape: {doc_embeddings.shape}")

    # Build dataset structure
    ys = [[d['category_idx']] for d in documents]

    return {
        'doc_texts': doc_texts,
        'ys': ys,
        'category_names': category_names,
        'category_embeddings': category_embeddings,
        'doc_embeddings': doc_embeddings,
        'n_categories': len(category_names),
        'n_documents': len(documents),
    }


def verify_diagnostics(dataset, T=5000):
    """Check all 3 axes deviate for a given horizon T."""
    from numpy.linalg import norm

    print(f"\n{'=' * 70}")
    print(f"VERIFYING DIAGNOSTICS (T={T})")
    print(f"{'=' * 70}")

    cat_emb = dataset['category_embeddings'].numpy()
    doc_emb = dataset['doc_embeddings'].numpy()
    ys = dataset['ys']
    K = dataset['n_categories']
    N = dataset['n_documents']

    # Normalize embeddings
    cat_norms = norm(cat_emb, axis=1, keepdims=True)
    cat_normed = cat_emb / (cat_norms + 1e-8)
    doc_norms = norm(doc_emb, axis=1, keepdims=True)
    doc_normed = doc_emb / (doc_norms + 1e-8)

    # --- AXIS B: Coherence ---
    print(f"\nAxis B (Coherence α̂):")
    cos_matrix = cat_normed @ cat_normed.T
    np.fill_diagonal(cos_matrix, 2.0)  # exclude self
    min_cos = cos_matrix.min(axis=1)
    alpha_hat = np.median(min_cos)
    print(f"  α̂ = {alpha_hat:.4f}")
    print(f"  Mean min-cosine = {min_cos.mean():.4f}")
    print(f"  Max min-cosine = {min_cos.max():.4f}")
    print(f"  Threshold: < 0.1 for deviating")
    axis_b = alpha_hat < 0.1
    print(f"  STATUS: {'DEVIATING ✓' if axis_b else 'SATURATED ✗'}")

    # --- AXIS A: Effective rank ---
    print(f"\nAxis A (Effective Rank r̂):")
    # Sample joint features φ(x,a) = x ⊙ a with random (x,a) pairs
    n_sample = min(10000, N)
    np.random.seed(42)
    sample_idx = np.random.choice(N, n_sample, replace=False)
    sample_arms = np.random.choice(K, n_sample, replace=True)

    phi_samples = doc_normed[sample_idx] * cat_normed[sample_arms]
    cov = np.cov(phi_samples.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = eigvals.sum() / eigvals.max()
    ratio = r_hat / np.sqrt(T)
    print(f"  r̂ = {r_hat:.2f}")
    print(f"  √T = {np.sqrt(T):.2f}")
    print(f"  r̂/√T = {ratio:.4f}")
    print(f"  Threshold: >= 1 for deviating")
    axis_a = ratio >= 1
    print(f"  STATUS: {'DEVIATING ✓' if axis_a else 'SATURATED ✗'}")
    # Suggest T for this to deviate
    T_threshold = r_hat ** 2
    print(f"  → Would deviate for T ≤ {T_threshold:.0f}")

    # --- AXIS C: Gap distribution ---
    print(f"\nAxis C (Gap Distribution):")
    from sklearn.linear_model import Ridge

    # Fit oracle model
    n_train = min(10000, N)
    train_idx = np.random.choice(N, n_train, replace=False)
    X_train = []
    y_train = []
    for idx in train_idx:
        true_cat = ys[idx][0]
        phi = doc_normed[idx] * cat_normed[true_cat]
        X_train.append(phi)
        y_train.append(1.0)
        # Negative examples (3 per positive for better model)
        for _ in range(3):
            neg_cat = np.random.choice(K)
            while neg_cat == true_cat:
                neg_cat = np.random.choice(K)
            phi_neg = doc_normed[idx] * cat_normed[neg_cat]
            X_train.append(phi_neg)
            y_train.append(0.0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    theta_oracle = ridge.coef_

    # Compute gaps
    n_test = min(2000, N)
    test_idx = np.random.choice(N, n_test, replace=False)
    gaps = []
    for idx in test_idx:
        scores = (doc_normed[idx:idx+1] * cat_normed) @ theta_oracle  # (K,)
        sorted_scores = np.sort(scores)[::-1]
        gap = sorted_scores[0] - sorted_scores[1]
        gaps.append(gap)

    gaps = np.array(gaps)
    # Use multiple thresholds
    for thresh in [0.001, 0.01, 0.05, 0.1]:
        frac = np.mean(gaps <= thresh)
        print(f"  Fraction with gap ≤ {thresh}: {frac:.4f}")

    frac_heavy = np.mean(gaps <= 0.01)
    axis_c = frac_heavy > 0.5
    print(f"  Threshold: > 0.5 for heavy tail (deviating)")
    print(f"  STATUS: {'DEVIATING ✓' if axis_c else 'SATURATED ✗'}")

    # Percentiles of gap distribution
    print(f"  Gap percentiles: p10={np.percentile(gaps,10):.4f}, p25={np.percentile(gaps,25):.4f}, p50={np.percentile(gaps,50):.4f}, p75={np.percentile(gaps,75):.4f}, p90={np.percentile(gaps,90):.4f}")

    # --- SUMMARY ---
    n_deviate = sum([axis_a, axis_b, axis_c])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {n_deviate}/3 axes deviate")
    if n_deviate >= 3:
        print("→ REGIME: warm-start-essential ★")
    elif n_deviate >= 2:
        print("→ REGIME: warm-start-likely")
    else:
        print("→ REGIME: LinUCB-suffices")
    print(f"{'=' * 70}")

    return {
        'alpha_hat': alpha_hat,
        'r_hat': r_hat,
        'r_over_sqrtT': ratio,
        'T_threshold': T_threshold,
        'gap_frac': frac_heavy,
        'n_deviate': n_deviate,
        'axis_a': axis_a,
        'axis_b': axis_b,
        'axis_c': axis_c,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING SYNTHETIC TEXT CLASSIFICATION DATASET v2")
    print(f"50 categories, 250 docs/category = 12,500 documents")
    print("=" * 70)

    # Generate documents
    documents, category_names = generate_documents_v2(n_per_category=250)
    print(f"\nTotal documents: {len(documents)}")
    print(f"Sample texts:")
    for i in range(3):
        print(f"  [{documents[i]['category_name']}] {documents[i]['text'][:100]}...")

    # Build dataset with embeddings
    dataset = build_dataset_v2(documents, category_names)

    # Try multiple T values
    for T in [500, 1000, 2000, 5000, 10000, 25000]:
        verify_diagnostics(dataset, T=T)

    # Save full dataset
    out_path = 'synthetic_text_dataset_v2.pkl.gz'
    print(f"\nSaving dataset to {out_path}...")
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Done! ({len(documents)} documents, {len(category_names)} categories)")
