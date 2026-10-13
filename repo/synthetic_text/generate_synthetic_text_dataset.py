"""
generate_synthetic_text_dataset.py

Creates a synthetic text classification dataset where:
- 200 categories spanning diverse domains (science, sports, arts, etc.)
- ~50,000 documents (250 per category)
- Categories are chosen to be semantically diverse → low coherence in embedding space
- Text is generated from templates with category-specific vocabulary

After generation:
1. Embeds categories + documents with sentence-transformers
2. Computes joint features φ(x,a) = x ⊙ a
3. Verifies all 3 diagnostic axes deviate

Output: synthetic_text_dataset.pkl.gz
"""

import numpy as np
import random
import gzip
import pickle
import torch
from collections import defaultdict

# ================================================================
# STEP 1: Define 200 diverse categories
# ================================================================
# We want categories from VERY different domains so their embeddings
# are near-orthogonal (low coherence). Organized by super-domain.

CATEGORIES = {
    # NATURAL SCIENCES (40)
    "Quantum Mechanics": ["quantum", "wave function", "entanglement", "superposition", "photon"],
    "Organic Chemistry": ["molecule", "synthesis", "carbon", "polymer", "reaction"],
    "Marine Biology": ["ocean", "coral reef", "fish species", "marine ecosystem", "plankton"],
    "Volcanology": ["volcano", "eruption", "magma", "lava flow", "tectonic"],
    "Paleontology": ["dinosaur", "fossil", "extinction", "Jurassic", "excavation"],
    "Astrophysics": ["black hole", "neutron star", "galaxy", "dark matter", "supernova"],
    "Microbiology": ["bacteria", "virus", "pathogen", "antibiotic", "cell culture"],
    "Meteorology": ["hurricane", "atmospheric pressure", "precipitation", "tornado", "climate"],
    "Botany": ["photosynthesis", "pollen", "root system", "flowering plant", "chlorophyll"],
    "Entomology": ["insect", "beetle", "pollinator", "larva", "exoskeleton"],
    "Genetics": ["DNA", "genome", "mutation", "heredity", "gene expression"],
    "Thermodynamics": ["entropy", "heat transfer", "thermal equilibrium", "Carnot cycle", "energy"],
    "Glaciology": ["glacier", "ice sheet", "permafrost", "moraine", "crevasse"],
    "Seismology": ["earthquake", "seismic wave", "fault line", "Richter scale", "aftershock"],
    "Crystallography": ["crystal lattice", "unit cell", "X-ray diffraction", "symmetry", "mineral"],
    "Oceanography": ["ocean current", "salinity", "deep sea", "hydrothermal vent", "tide"],
    "Nuclear Physics": ["radioactive decay", "fission", "isotope", "neutron", "half-life"],
    "Ecology": ["biodiversity", "food chain", "habitat", "conservation", "endangered species"],
    "Immunology": ["antibody", "immune response", "vaccine", "T-cell", "inflammation"],
    "Neuroscience": ["neuron", "synapse", "brain", "dopamine", "cognitive"],
    "Geomorphology": ["erosion", "sediment", "river delta", "canyon", "weathering"],
    "Parasitology": ["parasite", "host organism", "malaria", "tapeworm", "infection cycle"],
    "Acoustics": ["sound wave", "frequency", "resonance", "decibel", "vibration"],
    "Mycology": ["fungus", "mushroom", "spore", "mycelium", "decomposer"],
    "Fluid Dynamics": ["turbulence", "viscosity", "Reynolds number", "aerodynamics", "flow"],
    "Primatology": ["primate", "gorilla", "chimpanzee", "social behavior", "tool use"],
    "Spectroscopy": ["absorption spectrum", "wavelength", "photon emission", "infrared", "spectral"],
    "Limnology": ["freshwater lake", "algae bloom", "dissolved oxygen", "watershed", "aquatic"],
    "Cosmology": ["Big Bang", "cosmic inflation", "dark energy", "redshift", "universe expansion"],
    "Toxicology": ["poison", "dose-response", "carcinogen", "detoxification", "toxic exposure"],
    "Dendrochronology": ["tree ring", "annual growth", "dating method", "wood sample", "climate proxy"],
    "Ornithology": ["bird migration", "feather", "nest building", "avian species", "wingspan"],
    "Petrology": ["igneous rock", "metamorphic", "mineral composition", "granite", "basalt"],
    "Virology": ["viral replication", "RNA virus", "pandemic", "viral mutation", "host cell"],
    "Biophysics": ["protein folding", "molecular motor", "membrane potential", "ion channel", "lipid"],
    "Arachnology": ["spider", "web construction", "venom", "arachnid", "silk production"],
    "Pedology": ["soil composition", "humus", "clay mineral", "soil horizon", "nutrient cycle"],
    "Herpetology": ["reptile", "amphibian", "snake venom", "salamander", "cold-blooded"],
    "Atmospheric Chemistry": ["ozone layer", "greenhouse gas", "aerosol", "smog", "nitrogen oxide"],
    "Chronobiology": ["circadian rhythm", "biological clock", "melatonin", "sleep cycle", "photoperiod"],

    # SPORTS & ATHLETICS (30)
    "Basketball Strategy": ["pick and roll", "zone defense", "three-pointer", "fast break", "rebound"],
    "Marathon Running": ["endurance", "pace", "hydration", "finish line", "personal record"],
    "Olympic Swimming": ["freestyle stroke", "backstroke", "lap time", "pool lane", "diving"],
    "Cricket": ["wicket", "bowler", "innings", "batsman", "test match"],
    "Fencing": ["foil", "parry", "lunge", "touche", "piste"],
    "Rock Climbing": ["belay", "carabiner", "crux", "rappel", "bouldering"],
    "Archery": ["bow", "arrow", "quiver", "bullseye", "draw weight"],
    "Sumo Wrestling": ["dohyo", "yokozuna", "mawashi", "bout", "salt ritual"],
    "Polo": ["mallet", "chukker", "pony", "goal post", "handicap"],
    "Bobsled": ["sled", "ice track", "push start", "brakeman", "aerodynamic"],
    "Table Tennis": ["paddle", "spin serve", "backhand", "rally", "match point"],
    "Gymnastics": ["vault", "balance beam", "floor routine", "parallel bars", "dismount"],
    "Rowing": ["oar", "stroke rate", "regatta", "coxswain", "shell boat"],
    "Triathlon": ["swim leg", "bike segment", "transition zone", "ironman", "wetsuit"],
    "Curling": ["stone", "broom", "house", "skip", "ice sheet"],
    "Weightlifting": ["snatch", "clean and jerk", "barbell", "personal best", "platform"],
    "Surfing": ["wave", "board", "barrel", "wipeout", "swell"],
    "Badminton": ["shuttlecock", "smash", "net shot", "court", "racket"],
    "Rugby": ["scrum", "try", "lineout", "tackle", "conversion kick"],
    "Skiing": ["slalom", "mogul", "powder", "chairlift", "downhill"],
    "Ice Hockey": ["puck", "power play", "penalty box", "goaltender", "face-off"],
    "Volleyball": ["spike", "setter", "block", "dig", "rotation"],
    "Boxing": ["jab", "uppercut", "round", "knockout", "ring"],
    "Golf": ["driver", "fairway", "birdie", "putting green", "handicap"],
    "Cycling": ["peloton", "sprint finish", "mountain stage", "time trial", "velodrome"],
    "Judo": ["ippon", "throw", "groundwork", "belt rank", "tatami"],
    "Water Polo": ["goal", "treading water", "shot clock", "ejection", "cap"],
    "Skateboarding": ["ollie", "halfpipe", "kickflip", "grind", "deck"],
    "Sailing": ["tack", "jib", "hull", "windward", "keel"],
    "Lacrosse": ["crosse", "face-off", "crease", "ground ball", "clearing"],

    # ARTS & CULTURE (30)
    "Renaissance Painting": ["fresco", "perspective", "canvas", "Florentine", "oil paint"],
    "Jazz Music": ["improvisation", "saxophone", "swing rhythm", "blues scale", "ensemble"],
    "Kabuki Theater": ["actor", "costume", "stage", "dramatic pose", "traditional Japanese"],
    "Ceramic Pottery": ["kiln", "glaze", "clay wheel", "firing temperature", "stoneware"],
    "Ballet": ["pirouette", "pointe shoes", "arabesque", "corps de ballet", "choreography"],
    "Street Photography": ["candid", "urban", "black and white", "composition", "decisive moment"],
    "Opera": ["aria", "soprano", "libretto", "orchestra pit", "tenor"],
    "Woodblock Printing": ["carving", "ink", "impression", "woodcut", "relief"],
    "Flamenco": ["guitar", "dancer", "rhythm", "clapping", "passionate"],
    "Calligraphy": ["brush stroke", "ink", "script", "flourish", "parchment"],
    "Film Noir": ["detective", "shadow", "femme fatale", "cynical", "crime"],
    "Mosaic Art": ["tile", "pattern", "grout", "Byzantine", "tesserae"],
    "Hip Hop Production": ["beat", "sample", "turntable", "bass", "lyric"],
    "Origami": ["paper fold", "crane", "crease pattern", "square sheet", "sculpture"],
    "Stained Glass": ["window", "lead came", "cathedral", "colored light", "panel"],
    "Puppetry": ["marionette", "string", "stage", "character", "hand puppet"],
    "Tattoo Art": ["ink", "needle", "skin", "design", "parlor"],
    "Glassblowing": ["molten glass", "blowpipe", "furnace", "shaping", "annealing"],
    "Graffiti": ["spray paint", "wall", "tag", "mural", "stencil"],
    "Weaving": ["loom", "thread", "pattern", "shuttle", "textile"],
    "Mime": ["silent", "gesture", "invisible wall", "white face", "physical comedy"],
    "Sand Sculpture": ["beach", "carving", "water", "detail", "temporary"],
    "Ventriloquism": ["dummy", "voice throwing", "lip control", "comedy", "puppet"],
    "Ice Sculpture": ["chainsaw", "chisel", "frozen", "transparent", "melt"],
    "Mural Painting": ["large scale", "wall", "public art", "community", "brush"],
    "Beatboxing": ["vocal percussion", "rhythm", "mouth", "bass kick", "hi-hat"],
    "Shadow Puppetry": ["screen", "light source", "silhouette", "cutout", "story"],
    "Metal Sculpture": ["welding", "bronze", "abstract", "patina", "forge"],
    "Digital Animation": ["keyframe", "render", "character rig", "motion capture", "pixel"],
    "Tapestry": ["woven", "hanging", "narrative scene", "wool thread", "medieval"],

    # FOOD & COOKING (20)
    "Sushi Preparation": ["rice", "fish", "nori", "wasabi", "roll"],
    "French Pastry": ["croissant", "butter", "lamination", "choux", "patisserie"],
    "BBQ Smoking": ["brisket", "wood chips", "low and slow", "rub", "smoker"],
    "Bread Baking": ["sourdough", "yeast", "knead", "proof", "crust"],
    "Fermentation": ["kombucha", "sauerkraut", "culture", "probiotics", "jar"],
    "Chocolate Making": ["cacao", "tempering", "truffle", "cocoa butter", "mold"],
    "Winemaking": ["grape", "fermentation", "barrel", "vintage", "tannin"],
    "Thai Cuisine": ["lemongrass", "chili", "coconut milk", "pad thai", "fish sauce"],
    "Cheese Aging": ["cave", "rind", "mold", "curd", "affinage"],
    "Indian Spices": ["turmeric", "cumin", "cardamom", "masala", "curry"],
    "Molecular Gastronomy": ["spherification", "foam", "liquid nitrogen", "gel", "emulsion"],
    "Coffee Roasting": ["bean", "roast profile", "first crack", "origin", "grind"],
    "Pickle Making": ["brine", "cucumber", "vinegar", "jar", "preservation"],
    "Pizza Dough": ["Neapolitan", "oven", "stretching", "fermentation", "flour"],
    "Ice Cream": ["churning", "custard", "flavor", "freezer", "scoop"],
    "Dim Sum": ["dumpling", "steamer", "bamboo", "filling", "wrapper"],
    "Butchery": ["cut", "knife", "carcass", "marbling", "trimming"],
    "Maple Syrup": ["sap", "sugar shack", "boiling", "tree tap", "spring"],
    "Hot Sauce": ["pepper", "capsaicin", "vinegar", "Scoville", "fermented"],
    "Kombucha Brewing": ["SCOBY", "tea", "sugar", "second ferment", "carbonation"],

    # TECHNOLOGY & ENGINEERING (30)
    "Semiconductor Fabrication": ["wafer", "lithography", "etching", "cleanroom", "transistor"],
    "Rocket Propulsion": ["thrust", "combustion chamber", "nozzle", "oxidizer", "specific impulse"],
    "Blockchain": ["hash", "distributed ledger", "consensus", "smart contract", "mining"],
    "3D Printing": ["filament", "layer", "nozzle temperature", "bed adhesion", "slicing"],
    "Quantum Computing": ["qubit", "superposition", "gate", "decoherence", "entanglement"],
    "Autonomous Vehicles": ["lidar", "sensor fusion", "lane detection", "path planning", "obstacle"],
    "Wind Turbines": ["blade", "nacelle", "generator", "wind farm", "capacity factor"],
    "Desalination": ["reverse osmosis", "membrane", "saltwater", "brine", "freshwater"],
    "Bridge Engineering": ["suspension", "cable", "load bearing", "span", "foundation"],
    "Satellite Communication": ["orbit", "transponder", "uplink", "bandwidth", "ground station"],
    "Nuclear Reactor": ["coolant", "control rod", "fuel assembly", "containment", "chain reaction"],
    "Hydraulic Systems": ["piston", "fluid pressure", "cylinder", "pump", "valve"],
    "Textile Manufacturing": ["loom", "fiber", "dyeing", "spinning", "weave pattern"],
    "Oil Drilling": ["wellbore", "drill bit", "mud", "casing", "blowout preventer"],
    "Geothermal Energy": ["hot spring", "steam turbine", "underground heat", "well", "reservoir"],
    "Robotics": ["actuator", "servo motor", "end effector", "kinematics", "controller"],
    "Wastewater Treatment": ["sludge", "aeration", "settling tank", "effluent", "filtration"],
    "Fiber Optics": ["light pulse", "glass core", "cladding", "bandwidth", "splice"],
    "CNC Machining": ["spindle", "G-code", "tool path", "milling", "lathe"],
    "Solar Panel": ["photovoltaic", "silicon cell", "inverter", "panel array", "efficiency"],
    "Tunnel Boring": ["TBM", "shield", "segment", "excavation face", "alignment"],
    "Battery Technology": ["lithium ion", "anode", "cathode", "electrolyte", "charge cycle"],
    "Elevator Systems": ["cable", "counterweight", "shaft", "motor", "safety brake"],
    "Air Conditioning": ["compressor", "refrigerant", "evaporator", "condenser", "thermostat"],
    "Dam Construction": ["concrete", "spillway", "reservoir", "hydropower", "embankment"],
    "Radar Systems": ["antenna", "pulse", "echo", "doppler", "range"],
    "Submarine": ["hull pressure", "ballast tank", "periscope", "torpedo", "dive"],
    "Printing Press": ["ink roller", "plate", "impression cylinder", "paper feed", "offset"],
    "Crane Operation": ["boom", "hoist", "load chart", "outrigger", "rigging"],
    "Metal Casting": ["mold", "molten metal", "pouring", "sand casting", "foundry"],

    # HISTORY & SOCIETY (25)
    "Viking Exploration": ["longship", "Norse", "raid", "settlement", "saga"],
    "Ancient Egypt": ["pharaoh", "pyramid", "hieroglyph", "mummy", "Nile"],
    "Samurai Culture": ["katana", "bushido", "shogun", "feudal Japan", "honor"],
    "Industrial Revolution": ["steam engine", "factory", "coal", "urbanization", "mechanization"],
    "Silk Road Trade": ["caravan", "merchant", "spice", "oasis", "trade route"],
    "Roman Architecture": ["aqueduct", "arch", "colosseum", "concrete", "column"],
    "Aztec Civilization": ["temple", "sacrifice", "calendar", "Tenochtitlan", "cacao"],
    "Medieval Alchemy": ["philosopher's stone", "transmutation", "elixir", "crucible", "mercury"],
    "Piracy Golden Age": ["galleon", "treasure", "cutlass", "Jolly Roger", "Caribbean"],
    "Byzantine Empire": ["Constantinople", "mosaic", "emperor", "orthodox", "trade"],
    "Mongol Empire": ["cavalry", "steppe", "Genghis Khan", "conquest", "yurt"],
    "French Revolution": ["guillotine", "Bastille", "republic", "aristocracy", "liberty"],
    "Ottoman Empire": ["sultan", "janissary", "mosque", "conquest", "bazaar"],
    "Maya Civilization": ["temple pyramid", "calendar", "glyph", "cenote", "astronomy"],
    "Cold War Espionage": ["spy", "intelligence", "defector", "surveillance", "cipher"],
    "Prohibition Era": ["bootlegger", "speakeasy", "moonshine", "temperance", "gangster"],
    "Gold Rush": ["prospector", "pan", "nugget", "mining camp", "claim"],
    "Feudal System": ["lord", "vassal", "fief", "serf", "manor"],
    "Inuit Traditions": ["igloo", "kayak", "seal hunting", "tundra", "aurora"],
    "Polynesian Navigation": ["star compass", "canoe", "wave pattern", "island", "voyage"],
    "Medieval Siege": ["catapult", "castle wall", "battering ram", "moat", "tower"],
    "Gladiator Combat": ["arena", "trident", "shield", "emperor's judgment", "chariot"],
    "Transcontinental Railroad": ["locomotive", "spike", "track laying", "tunnel", "expansion"],
    "Whaling Industry": ["harpoon", "blubber", "whale oil", "ship", "processing"],
    "Lighthouse Keeping": ["beacon", "foghorn", "lens", "cliff", "maritime"],

    # NATURE & GEOGRAPHY (25)
    "Coral Reef Ecosystem": ["polyp", "symbiosis", "bleaching", "tropical fish", "atoll"],
    "Arctic Tundra": ["permafrost", "caribou", "lichen", "polar bear", "frozen"],
    "Rainforest Canopy": ["epiphyte", "canopy layer", "biodiversity", "vine", "humidity"],
    "Desert Survival": ["cactus", "oasis", "sand dune", "heat", "water conservation"],
    "Volcanic Island": ["caldera", "lava tube", "hot spring", "basalt", "eruption"],
    "Cave Systems": ["stalactite", "stalagmite", "underground river", "cavern", "limestone"],
    "Mangrove Forest": ["root system", "brackish water", "nursery habitat", "coast", "salt tolerant"],
    "Alpine Meadow": ["wildflower", "elevation", "treeline", "grazing", "snowmelt"],
    "Kelp Forest": ["giant kelp", "sea otter", "underwater", "holdfast", "canopy"],
    "Wetland Marsh": ["cattail", "waterfowl", "peat", "flooding", "biodiversity"],
    "Sand Desert": ["erg", "wind erosion", "dune migration", "scorpion", "nomad"],
    "Fjord Formation": ["glacial carving", "steep cliff", "inlet", "Norway", "deep water"],
    "Savanna Grassland": ["acacia tree", "grazer", "dry season", "wildebeest", "fire"],
    "Peat Bog": ["sphagnum", "acidic water", "decomposition", "carbon sink", "preservation"],
    "River Estuary": ["mixing zone", "sediment", "tidal", "nursery ground", "brackish"],
    "Barrier Island": ["sand bar", "wave action", "coastal protection", "dune", "inlet"],
    "Geothermal Field": ["geyser", "fumarole", "hot mud", "sulfur", "thermal pool"],
    "Bamboo Forest": ["rapid growth", "panda habitat", "culm", "rhizome", "dense"],
    "Salt Flat": ["evaporation", "mineral crust", "flat expanse", "lithium", "dry lake"],
    "Underwater Volcano": ["seamount", "hydrothermal", "pillow lava", "mid-ocean ridge", "vent"],
    "Temperate Rainforest": ["moss", "old growth", "fern", "rainfall", "evergreen"],
    "Prairie Grassland": ["bison", "tall grass", "root depth", "fire ecology", "horizon"],
    "Karst Landscape": ["sinkhole", "limestone dissolution", "underground", "cave", "spring"],
    "Tidal Pool": ["sea anemone", "starfish", "low tide", "rocky shore", "crustacean"],
    "Cloud Forest": ["mist", "epiphyte", "high altitude", "moss-covered", "fog"],
}

assert len(CATEGORIES) == 200, f"Expected 200 categories, got {len(CATEGORIES)}"


# ================================================================
# STEP 2: Generate text documents
# ================================================================

TEMPLATES = [
    "Title: {topic} Research\nContent: A study on {kw1} and {kw2} reveals new findings about {kw3} in the context of {topic}.",
    "Title: Understanding {kw1}\nContent: This article explores how {kw1} relates to {kw2} and {kw3} in {topic}.",
    "Title: Advances in {topic}\nContent: Recent developments in {kw1} have led to breakthroughs involving {kw2} and {kw3}.",
    "Title: The Role of {kw2} in {topic}\nContent: Experts examine {kw1}, {kw2}, and {kw3} to better understand {topic}.",
    "Title: {topic}: A Comprehensive Overview\nContent: {kw1} and {kw2} are central concepts in {topic}, along with {kw3}.",
    "Title: New Perspectives on {kw3}\nContent: In the field of {topic}, {kw3} interacts with {kw1} and {kw2} in complex ways.",
    "Title: {kw1} and {kw2}\nContent: The relationship between {kw1} and {kw2} is fundamental to {topic}, especially regarding {kw3}.",
    "Title: Practical Applications of {topic}\nContent: {kw1} technology combined with {kw2} enables new approaches to {kw3}.",
    "Title: {topic} for Beginners\nContent: An introduction to {kw1}, {kw2}, and {kw3} — the building blocks of {topic}.",
    "Title: Challenges in {topic}\nContent: Despite advances in {kw1}, issues with {kw2} and {kw3} remain unsolved in {topic}.",
]


def generate_documents(n_per_category=250, seed=42):
    """Generate text documents for all categories."""
    random.seed(seed)
    np.random.seed(seed)

    documents = []  # list of (title, content, category_idx, category_name)
    category_names = list(CATEGORIES.keys())

    for cat_idx, cat_name in enumerate(category_names):
        keywords = CATEGORIES[cat_name]
        for _ in range(n_per_category):
            template = random.choice(TEMPLATES)
            kws = random.sample(keywords, min(3, len(keywords)))
            while len(kws) < 3:
                kws.append(random.choice(keywords))

            text = template.format(topic=cat_name, kw1=kws[0], kw2=kws[1], kw3=kws[2])
            # Split into title and content
            parts = text.split("\nContent: ")
            title = parts[0].replace("Title: ", "")
            content = parts[1] if len(parts) > 1 else ""

            documents.append({
                'title': title,
                'content': content,
                'category_idx': cat_idx,
                'category_name': cat_name,
            })

    # Shuffle
    random.shuffle(documents)
    return documents, category_names


def build_dataset(documents, category_names, st_model_name='sentence-transformers/bert-base-nli-mean-tokens'):
    """Embed everything and build the bandit dataset."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading SentenceTransformer: {st_model_name}")
    st_model = SentenceTransformer(st_model_name)

    # Embed category names (arm embeddings)
    print(f"Embedding {len(category_names)} category names...")
    category_embeddings = st_model.encode(category_names, batch_size=64, show_progress_bar=True)
    category_embeddings = torch.FloatTensor(category_embeddings)
    print(f"  Category embeddings shape: {category_embeddings.shape}")

    # Embed documents (context embeddings)
    print(f"Embedding {len(documents)} documents...")
    doc_texts = [f"{d['title']}. {d['content']}" for d in documents]
    doc_embeddings = st_model.encode(doc_texts, batch_size=256, show_progress_bar=True)
    doc_embeddings = torch.FloatTensor(doc_embeddings)
    print(f"  Document embeddings shape: {doc_embeddings.shape}")

    # Build dataset structure
    text_titles = [d['title'] for d in documents]
    text_contents = [d['content'] for d in documents]
    ys = [[d['category_idx']] for d in documents]  # list of lists (matching existing format)

    # Joint feature: phi(x,a) = x ⊙ a
    # Context = doc embedding, Arms = category embeddings
    # Xs stores [context_embedding | zeros] (matching existing format where first half is context)
    d_emb = doc_embeddings.shape[1]
    Xs = torch.zeros(len(documents), d_emb * 2)
    Xs[:, :d_emb] = doc_embeddings

    # labelfeats: {label_name: (idx, embedding)}
    labelfeats = {}
    for i, name in enumerate(category_names):
        labelfeats[name] = (i, category_embeddings[i].numpy())

    return {
        'Xs': Xs,
        'ys': ys,
        'text_titles': text_titles,
        'text_contents': text_contents,
        'labelfeats': labelfeats,
        'category_names': category_names,
        'category_embeddings': category_embeddings,
        'doc_embeddings': doc_embeddings,
        'n_categories': len(category_names),
        'n_documents': len(documents),
    }


def verify_diagnostics(dataset):
    """Check all 3 axes deviate."""
    print("\n" + "=" * 70)
    print("VERIFYING DIAGNOSTICS")
    print("=" * 70)

    cat_emb = dataset['category_embeddings'].numpy()
    doc_emb = dataset['doc_embeddings'].numpy()
    ys = dataset['ys']
    K = len(dataset['category_names'])
    N = len(ys)
    T = N  # horizon = number of samples

    # Normalize category embeddings
    from numpy.linalg import norm
    cat_norms = norm(cat_emb, axis=1, keepdims=True)
    cat_normed = cat_emb / (cat_norms + 1e-8)

    # --- AXIS B: Coherence ---
    print(f"\nAxis B (Coherence):")
    cos_matrix = cat_normed @ cat_normed.T
    np.fill_diagonal(cos_matrix, 2.0)  # exclude self
    min_cos = cos_matrix.min(axis=1)
    alpha_hat = np.median(min_cos)
    print(f"  α̂ = {alpha_hat:.4f}")
    print(f"  Threshold: < 0.1 for deviating")
    print(f"  STATUS: {'DEVIATING ✓' if alpha_hat < 0.1 else 'SATURATED ✗'}")

    # --- AXIS A: Effective rank ---
    print(f"\nAxis A (Effective Rank):")
    # Compute joint features for a sample
    n_sample = min(5000, N)
    np.random.seed(42)
    sample_idx = np.random.choice(N, n_sample, replace=False)
    sample_arms = np.random.choice(K, n_sample, replace=True)

    # Normalize doc embeddings too
    doc_norms = norm(doc_emb, axis=1, keepdims=True)
    doc_normed = doc_emb / (doc_norms + 1e-8)

    phi_samples = doc_normed[sample_idx] * cat_normed[sample_arms]
    cov = np.cov(phi_samples.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = eigvals.sum() / eigvals.max()
    ratio = r_hat / np.sqrt(T)
    print(f"  r̂ = {r_hat:.2f}")
    print(f"  r̂/√T = {ratio:.4f} (T={T})")
    print(f"  Threshold: >= 1 for deviating")
    print(f"  STATUS: {'DEVIATING ✓' if ratio >= 1 else 'SATURATED ✗'}")

    # --- AXIS C: Gap distribution ---
    print(f"\nAxis C (Gap Distribution):")
    # Fit oracle model and check gap distribution
    # Simple approach: compute oracle gaps from true labels
    # For each sample, gap = reward(best arm) - reward(2nd best arm)
    # With binary reward, gap is about how many arms tie
    # With 200 arms and 1 correct answer, most gaps are 0 (since most arms give 0 reward)
    # Actually with binary rewards (1 if correct, 0 otherwise), the gap is always 1
    # (best arm gives reward 1, all others give 0)
    # BUT in the linear model with φ(x,a) = x⊙a and θ*, the expected reward varies continuously
    
    # Let's fit a ridge model
    print("  Fitting oracle ridge model...")
    # Use all samples with their true labels
    from sklearn.linear_model import Ridge
    
    # Build training data: for each sample, the correct arm's phi gives reward 1
    n_train = min(10000, N)
    train_idx = np.random.choice(N, n_train, replace=False)
    X_train = []
    y_train = []
    for idx in train_idx:
        true_cat = ys[idx][0]
        phi = doc_normed[idx] * cat_normed[true_cat]
        X_train.append(phi)
        y_train.append(1.0)
        # Add a negative example
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

    # Compute gaps for test samples
    n_test = min(5000, N)
    test_idx = np.random.choice(N, n_test, replace=False)
    gaps = []
    for idx in test_idx:
        scores = []
        for a in range(K):
            phi = doc_normed[idx] * cat_normed[a]
            scores.append(theta_oracle @ phi)
        scores = np.array(scores)
        sorted_scores = np.sort(scores)[::-1]
        gap = sorted_scores[0] - sorted_scores[1]
        gaps.append(gap)

    gaps = np.array(gaps)
    frac_small = np.mean(gaps <= 0.01)
    print(f"  Fraction with gap <= 0.01: {frac_small:.4f}")
    print(f"  Threshold: > 0.5 for heavy tail (deviating)")
    print(f"  STATUS: {'DEVIATING ✓' if frac_small > 0.5 else 'SATURATED ✗'}")

    # --- SUMMARY ---
    n_deviate = sum([alpha_hat < 0.1, ratio >= 1, frac_small > 0.5])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {n_deviate}/3 axes deviate")
    if n_deviate >= 3:
        print("REGIME: warm-start-essential ★")
    elif n_deviate >= 2:
        print("REGIME: warm-start-likely")
    else:
        print("REGIME: LinUCB-suffices")
    print(f"{'=' * 70}")

    return {
        'alpha_hat': alpha_hat,
        'r_hat': r_hat,
        'r_over_sqrtT': ratio,
        'gap_frac': frac_small,
        'n_deviate': n_deviate,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING SYNTHETIC TEXT CLASSIFICATION DATASET")
    print("=" * 70)

    # Generate documents
    print(f"\nGenerating documents for {len(CATEGORIES)} categories...")
    documents, category_names = generate_documents(n_per_category=250)
    print(f"  Total documents: {len(documents)}")
    print(f"  Sample: {documents[0]['title'][:60]}... [{documents[0]['category_name']}]")

    # Build dataset with embeddings
    print("\nBuilding dataset with embeddings...")
    dataset = build_dataset(documents, category_names)

    # Verify diagnostics
    diag = verify_diagnostics(dataset)

    # Save
    out_path = 'synthetic_text_dataset.pkl.gz'
    print(f"\nSaving dataset to {out_path}...")
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Done! ({len(documents)} documents, {len(category_names)} categories)")
