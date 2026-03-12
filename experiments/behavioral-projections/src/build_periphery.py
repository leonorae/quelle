"""Build periphery probes corpus component (D12).

Generates formally diverse, out-of-distribution, and unusual inputs designed
to activate low-density regions of the model's representational geometry.

    python -m src.build_periphery [--output-dir prompts] [--seed 42]

Produces: periphery_probes.jsonl (~600 prompts)
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path


def _make_malformed(rng: random.Random) -> list[dict]:
    """Truncated, garbled, typo-ridden prompts."""
    prompts = []

    # Truncated mid-sentence
    truncated = [
        "The fundamental problem with quantum mechanics is that",
        "In order to understand why the economy",
        "She walked into the room and immediately noticed",
        "The algorithm works by first computing the",
        "According to the latest research on climate",
        "When you consider the implications of",
        "The patient presented with symptoms of",
        "If we assume that the initial conditions are",
        "The committee voted to approve the",
        "During the Renaissance, artists began to",
        "The key difference between TCP and",
        "In Bayesian statistics, the posterior is",
        "The main advantage of using a hash",
        "Consider a scenario where two trains",
        "The function returns None when",
    ]

    # Garbled / typo-heavy
    garbled = [
        "Waht si teh captial of Frnace?",
        "Expalin howw photosytnhesis wroks in plnats.",
        "Hw mny plnets r n th slr systm?",
        "Descirbe the proccess of eveloution by natrual sleection.",
        "Whta are teh main differnces betewen RNA adn DNA?",
        "Caluclate the derviatvie of x sqaured puls three x.",
        "Exlpain teh concpet of entorpy in thermodyanmics.",
        "Waht is teh porpose of a compilre?",
        "Dscribe hw a neruon trnasmits signlas.",
        "Wht r th prprts f prm nmbrs?",
    ]

    # Repeated words / stuttering
    repeated = [
        "What what what is the meaning of of of life?",
        "The the the cat sat on on the the mat.",
        "Please please explain explain quantum quantum computing computing.",
        "How how does does gravity gravity work work?",
        "I I I need need to to understand understand this this.",
    ]

    # Missing words (grammatically broken)
    missing = [
        "What capital France?",
        "Explain works photosynthesis.",
        "How many in solar system?",
        "Describe process evolution selection.",
        "Calculate derivative squared plus three.",
        "The important thing about is that it.",
        "When you the result is always.",
        "If then but otherwise not.",
        "She the and then it was.",
        "They went to the and found.",
    ]

    for i, text in enumerate(truncated + garbled + repeated + missing):
        prompts.append({
            "prompt_id": f"periph_malformed_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "malformed",
            "group_id": None,
        })

    return prompts


def _make_mixed_language(rng: random.Random) -> list[dict]:
    """Code-switching, non-English, transliteration."""
    prompts = []

    mixed = [
        # Code-switching
        "What is the bedeutung of this Konzept in der modernen Physik?",
        "Explain el concepto de la gravedad in simple terms por favor.",
        "この問題について what do you think? Please explain in English.",
        "Wie funktioniert photosynthesis exactly? Ich verstehe das nicht.",
        "Can you expliquer comment les réseaux neuronaux work?",
        "Что такое entropy and why is it важно for thermodynamics?",
        "Explain 递归 recursion como funciona in computer science.",
        "Was ist der Unterschied between RNA и DNA?",

        # Pure non-English (Pythia saw some but not much)
        "Quelle est la capitale de la France?",
        "Was ist der Sinn des Lebens?",
        "¿Cómo funciona la fotosíntesis?",
        "Что такое квантовая механика?",
        "量子力学とは何ですか？",
        "광합성은 어떻게 작동합니까?",
        "كيف تعمل الجاذبية؟",
        "Como funciona a gravidade?",

        # Transliteration / romanization
        "Nani ga quantum mechanics desu ka?",
        "Kvantovaya mekhanika - eto chto?",
        "Photosynthese wa dou hataraku no?",
        "Shenme shi liangsuo lilun?",

        # Emoji-heavy
        "🔬 Explain 🧬 DNA → RNA → 🧪 protein 🔄 please 🙏",
        "What is 🌍 + ☀️ + 🌱 = ??? in science terms",
        "⚛️ → 💥 → 🌟 explain this process",

        # Mixed script
        "Explain квантовая entanglement で お願いします",
        "The 道 of programming: 如何 to write clean code?",

        # Leetspeak / internet dialect
        "h0w d03s phot0synth3s1s w0rk l0l",
        "pls xplain y gravity iz a thing thx",
        "eli5 quantum stuff plz kthx",
        "wat iz entropy n y shud i care bout it",
    ]

    for i, text in enumerate(mixed):
        prompts.append({
            "prompt_id": f"periph_mixed_lang_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "mixed_language",
            "group_id": None,
        })

    return prompts


def _make_contradictory(rng: random.Random) -> list[dict]:
    """Self-contradictory or impossible prompts."""
    prompts = []

    contradictions = [
        "Explain why water is dry.",
        "Describe the color of a transparent mirror in complete darkness.",
        "What is the sound of silence in a vacuum?",
        "Calculate the square root of negative one using only real numbers.",
        "Explain how to travel faster than light without violating physics.",
        "Describe what happened before the beginning of time.",
        "Why is the number 7 both even and odd?",
        "Explain the taste of the color blue.",
        "How much does a thought weigh in kilograms?",
        "Describe the north pole of a sphere that has no poles.",
        "What temperature is absolute hot?",
        "Explain why 1 equals 2.",
        "Describe the shape of a four-sided triangle.",
        "What is the last digit of pi?",
        "Explain how to divide by zero and get a finite answer.",
        "Why do objects fall upward in normal gravity?",
        "What happened after the end of infinity?",
        "Describe the smell of a number.",
        "How do you measure the width of a point?",
        "What is north of the North Pole?",
        "Explain why all prime numbers are even.",
        "Describe a circle with exactly three corners.",
        "What is the chemical formula of nothing?",
        "How do you count to infinity and what is the last number?",
        "Explain the texture of empty space.",
        "Why is absolute zero hot?",
        "What is the opposite of existence?",
        "Describe the weight of a shadow.",
        "How bright is pure darkness?",
        "What is the past tense of the future?",
    ]

    for i, text in enumerate(contradictions):
        prompts.append({
            "prompt_id": f"periph_contradict_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "contradictory",
            "group_id": None,
        })

    return prompts


def _make_nonsense(rng: random.Random) -> list[dict]:
    """High-perplexity, random, adversarial strings."""
    prompts = []

    # Hand-crafted nonsense with grammatical structure
    structured_nonsense = [
        "The purple algorithm danced through the fibonacci sunset while calculating emotions.",
        "Seven abstract butterflies compiled the recursive ocean into a stack of metaphors.",
        "The eigenvalue of Tuesday exceeded the categorical imperative by three semicolons.",
        "Colorless green ideas sleep furiously in the gradient descent of consciousness.",
        "The monad lifted itself by its own bootstraps through a functor of despair.",
        "An isomorphism between breakfast and topology revealed the hidden curry in the proof.",
        "The garbage collector freed seventeen memories of future events that hadn't been allocated.",
        "A polymorphic duck typed its way through the existential quantifier of a lazy evaluation.",
        "The residual stream of consciousness overflowed into the attention sink of Tuesday.",
        "Three monads walked into a bar and composed themselves into an applicative functor.",
    ]

    # Random token sequences (pseudo-random but deterministic)
    random_sequences = []
    words = [
        "cat", "blue", "seven", "through", "quickly", "the", "and", "but",
        "compile", "electron", "yesterday", "if", "matrix", "dance", "heavy",
        "above", "nothing", "recursive", "sweet", "gradient", "fold", "between",
        "almost", "singular", "beneath", "triangle", "forget", "parallel",
        "quantum", "bread", "inverse", "hollow", "migrate", "tensor",
    ]
    for i in range(20):
        seq_rng = random.Random(42 + i)
        length = seq_rng.randint(8, 20)
        seq = " ".join(seq_rng.choice(words) for _ in range(length))
        random_sequences.append(seq)

    # Pure character-level noise
    char_noise = []
    for i in range(10):
        noise_rng = random.Random(100 + i)
        length = noise_rng.randint(30, 80)
        chars = string.ascii_letters + string.digits + " " * 5 + ".,;:!?"
        noise = "".join(noise_rng.choice(chars) for _ in range(length))
        char_noise.append(noise)

    # Adversarial-ish patterns
    adversarial = [
        "A" * 200,
        " ".join(["the"] * 50),
        "?" * 100,
        "\n".join(["line"] * 30),
        "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 4,
        "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1",
        "( ( ( ( ( ( ( ( ( ( ) ) ) ) ) ) ) ) ) )",
        "{ key: { key: { key: { key: { key: value } } } } }",
        "<tag><tag><tag><tag></tag></tag></tag></tag>",
        "SELECT * FROM table WHERE column = 'value' OR 1=1; DROP TABLE--",
    ]

    all_nonsense = structured_nonsense + random_sequences + char_noise + adversarial
    for i, text in enumerate(all_nonsense):
        prompts.append({
            "prompt_id": f"periph_nonsense_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "nonsense",
            "group_id": None,
        })

    return prompts


def _make_unusual_registers(rng: random.Random) -> list[dict]:
    """Same-ish semantics, radically different form."""
    prompts = []

    registers = [
        # Poetry
        "O photosynthesis! thou green miracle,\nBy which the leaf doth drink the sun's own fire,\nAnd from the air extract its sweetest particle,\nTo build the sugar that all life require.",
        "gravity pulls / everything down / even light / bends around / massive things",
        "roses are red\nviolets are blue\nentropy increases\nand so does the queue",

        # Legalese
        "WHEREAS the party of the first part (hereinafter 'the photon') shall be deemed to exhibit properties of both wave and particle nature; AND WHEREAS such duality shall not be construed as a contradiction; NOW THEREFORE the observer agrees to measure only one property at a time.",
        "The undersigned hereby acknowledges that the force of gravity, as defined in Newton's Universal Law of Gravitation (F = Gm₁m₂/r²), shall apply uniformly to all parties regardless of mass, composition, or jurisdiction.",

        # IRC / chat log
        "<user42> hey anyone know how photosynthesis works\n<botanist> basically plants eat sunlight lol\n<user42> no but like the actual chemistry\n<botanist> ok so chlorophyll absorbs photons right\n<user42> go on\n<botanist> then it splits water molecules\n<user42> wild",
        "user: can someone explain quantum entanglement\nmod: its when two particles are linked\nuser: but HOW\nmod: nobody really knows tbh\nuser: great thanks",

        # Git commit messages
        "fix: resolve race condition in photosynthesis light reaction\n\nThe electron transport chain was not properly synchronized\nwith the Calvin cycle, leading to occasional NADPH starvation\nunder high-light conditions.\n\nFixes #42",
        "refactor(gravity): simplify force calculation\n\nREAKING CHANGE: removed support for Newtonian gravity\nin favor of general relativity. Flat spacetime users\nshould pin to v1.x.",

        # Recipe format
        "QUANTUM ENTANGLEMENT\nServes: 2 particles\nPrep time: instantaneous\nIngredients:\n- 2 correlated particles\n- 1 shared quantum state\n- measurement apparatus\nDirections:\n1. Prepare particles in entangled state\n2. Separate to arbitrary distance\n3. Measure one particle\n4. Other particle collapses immediately\nNote: Does not transmit information faster than light.",

        # Stage directions
        "[Enter GRAVITY, stage left. MASS stands center stage, looking confused.]\nGRAVITY: Come hither, Mass. I bend the very fabric upon which thou stand'st.\nMASS: But how? I feel thy pull, yet see thee not.\nGRAVITY: 'Tis not a force, but geometry itself. The stage curves beneath thy feet.\n[SPACETIME ripples. Lights dim.]",

        # Academic abstract
        "Abstract: We present a novel investigation into the thermodynamic properties of making a cup of tea. Using a randomized controlled trial (n=47 cups), we demonstrate that water temperature at time of pouring (T₀) is the primary predictor of steep quality (p<0.001). Milk-first vs milk-last showed no significant effect (p=0.73), contradicting prior work by Smith et al. (2019).",

        # Medical case report style
        "CHIEF COMPLAINT: Patient presents with inability to understand quantum mechanics.\nHISTORY OF PRESENT ILLNESS: 35 y/o graduate student with 3-year history of progressive confusion regarding wave function collapse. Symptoms worsen when reading about the measurement problem. No relief from Copenhagen interpretation.\nASSESSMENT: Chronic quantum bewilderment, likely exacerbated by Many-Worlds exposure.",

        # Regex / formal language
        "^(?:photosynthesis|respiration)\\s+(?:in|of)\\s+(?:plants|cells)\\s*(?:\\(.*?\\))?$",
        "S → NP VP\nNP → Det N | Det N PP\nVP → V NP | V NP PP\nDet → 'the' | 'a'\nN → 'photon' | 'electron' | 'wave'\nV → 'absorbed' | 'emitted' | 'scattered'\nPP → P NP\nP → 'by' | 'from' | 'through'",

        # Bureaucratic form
        "FORM 7B-QUANTUM: REQUEST FOR PARTICLE STATE OBSERVATION\nSection 1: Observer Information\nName: ________________\nInstitution: ________________\nSection 2: Particle Details\nType: □ Photon □ Electron □ Other: ____\nCurrent State: □ Superposition □ Collapsed □ Unknown\nSection 3: Measurement Basis\n□ Position □ Momentum □ Spin\nWARNING: Selecting both position and momentum may violate Heisenberg Policy §3.14",

        # Tweet thread
        "1/ ok let me explain why entropy matters for literally everything 🧵\n2/ so basically entropy = disorder right? WRONG. entropy = number of microstates\n3/ your room getting messy isn't entropy. well it is. but that's not the POINT\n4/ the real thing is: there are WAY more disordered states than ordered ones\n5/ so statistically, things just... tend toward disorder. not because of a force. because of MATH",

        # Haiku sequence
        "Electron jumps high\nPhoton released in the fall\nLight from quantum leap\n\nGravity bends space\nMass tells geometry how\nTo curve around it\n\nEntropy increases\nOrder dissolves into noise\nTime's arrow flies on",

        # Stack trace
        "Traceback (most recent call last):\n  File \"universe.py\", line 1, in <module>\n    from physics import gravity\n  File \"physics.py\", line 42, in <module>\n    force = G * m1 * m2 / r**2\nZeroDivisionError: division by zero\n# Note: r=0 when two point masses occupy the same location\n# This is why we need quantum gravity",

        # Cooking blog preamble style
        "When I was growing up in a small town in Ohio, my grandmother used to tell me stories about thermodynamics. I remember sitting on her porch, watching the sunset, and thinking about how heat always flows from hot to cold. That memory inspired me to write this explanation of the second law of thermodynamics. But first, let me tell you about the time I visited a power plant in 1987...",
    ]

    for i, text in enumerate(registers):
        prompts.append({
            "prompt_id": f"periph_register_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "unusual_register",
            "group_id": None,
        })

    return prompts


def _make_naturalistic_prose(rng: random.Random) -> list[dict]:
    """Plain prose — not questions. Fills the actual center of Pythia's
    training distribution, which our question-heavy corpus undersells."""
    prompts = []

    prose = [
        "The city had changed since he'd last visited. New buildings rose where old ones had stood, and the streets seemed narrower somehow, crowded with people he didn't recognize. He walked toward the river, hoping it at least would be the same.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll molecules in the thylakoid membranes of chloroplasts. The light-dependent reactions produce ATP and NADPH, which then drive the Calvin cycle.",
        "The committee met on Thursday to discuss the proposed changes to the zoning regulations. Several residents spoke in opposition, citing concerns about increased traffic and the impact on property values. The vote was tabled until the next meeting.",
        "She opened the package carefully, peeling back layers of brown paper to reveal a small wooden box. Inside, wrapped in tissue, was a pocket watch — tarnished but still ticking. There was no note, no return address.",
        "In supervised learning, the model is trained on labeled examples where each input is paired with a desired output. The training process adjusts the model's parameters to minimize the difference between its predictions and the true labels, typically measured by a loss function such as cross-entropy or mean squared error.",
        "The rain started just after noon, light at first, then heavier. By three o'clock the gutters were overflowing and the parking lot had become a shallow lake. Nobody left the building until it stopped around five.",
        "Mercury is the smallest planet in the solar system and the closest to the Sun. Its surface temperature ranges from -180°C at night to 430°C during the day. It has no atmosphere to speak of and no moons.",
        "He tried the door. Locked. He tried the window — also locked, but the latch was loose. With some effort he worked it free and pushed the window open. The room inside was dark and smelled of dust.",
        "The algorithm maintains a priority queue of nodes sorted by their estimated distance to the goal. At each step, it removes the node with the lowest estimated total cost, expands it, and adds its neighbors to the queue if they haven't been visited or if a shorter path to them has been found.",
        "Tuesday was market day. The square filled with stalls selling vegetables, cheese, bread, and fish. An old woman sold herbs from a basket. Children ran between the stalls while their parents haggled over prices.",
        "The Federal Reserve raised interest rates by 25 basis points at its March meeting, citing persistent inflationary pressures. Markets had largely priced in the move, and the S&P 500 closed essentially flat on the day.",
        "Water flows downhill. This simple fact explains rivers, erosion, flooding, and much of the shape of the landscape. It also explains why civilizations tend to build in valleys and why aqueducts were among the most important engineering achievements of the ancient world.",
        "The patient was a 67-year-old male presenting with chest pain radiating to the left arm, onset approximately two hours prior to arrival. ECG showed ST elevation in leads II, III, and aVF. Troponin levels were elevated. The cardiology team was consulted.",
        "She finished her coffee, rinsed the mug, and set it upside down on the drying rack. Through the kitchen window she could see the garden — overgrown now, after weeks of rain. The roses needed pruning.",
        "A binary search tree is a data structure where each node has at most two children. For any given node, all values in the left subtree are less than the node's value, and all values in the right subtree are greater. This property enables efficient search, insertion, and deletion operations.",
        "The expedition set out from base camp at dawn on the fourth day. The weather had cleared overnight and the summit was visible for the first time since their arrival. They estimated eight hours to the top, assuming conditions held.",
        "Congress passed the bill with bipartisan support, though several amendments were stripped during conference. The president signed it into law the following week. Implementation was delegated to the Department of Commerce, with a compliance deadline of eighteen months.",
        "He had been writing the same paper for three years. Every time he thought it was finished, he found another problem — a gap in the proof, an unclear definition, a result that contradicted something on page forty-seven.",
        "The voltage across a resistor is proportional to the current flowing through it. This relationship, known as Ohm's law, is written V = IR, where V is voltage in volts, I is current in amperes, and R is resistance in ohms.",
        "They sat across from each other at the small table, not speaking. The candle between them flickered. Outside, a car passed, its headlights sweeping across the wall. She looked at her hands. He looked at the door.",
        "The soil in this region is predominantly clay, which retains moisture well but drains poorly. This makes it suitable for rice cultivation but challenging for most root vegetables. Local farmers have adapted by building raised beds and adding organic matter to improve drainage.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        "The train was late. It was always late on Mondays. The platform was crowded with commuters checking their phones, adjusting their headphones, staring at the tracks. A pigeon walked along the yellow line as if it had somewhere important to be.",
        "In 1928, Alexander Fleming noticed that a mold growing on one of his petri dishes had killed the surrounding bacteria. This accidental observation led to the discovery of penicillin, which would go on to save millions of lives and transform the practice of medicine.",
        "The function accepts a list of integers and returns a dictionary mapping each unique value to its frequency. Edge cases include an empty list (returns empty dict) and a list with all identical elements (returns single-entry dict).",
    ]

    for i, text in enumerate(prose):
        prompts.append({
            "prompt_id": f"periph_prose_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "naturalistic_prose",
            "group_id": None,
        })

    return prompts


def _make_domain_outliers(rng: random.Random) -> list[dict]:
    """Highly specialized, archaic, or technically dense text."""
    prompts = []

    outliers = [
        # Archaic English
        "Wherefore doth the apple fall, and not ascend? For the Earth doth pull upon all bodies with a force proportional to their masse, as Newton hath demonstrated.",
        "In sooth, the particle doth exist in superposition, being both here and not here, until such time as observation doth collapse the wave function.",
        "Hear ye, hear ye! Let it be known throughout the realm that the second law of thermodynamics doth decree that entropy shall not decrease in an isolated system.",

        # Dense mathematical notation (as text)
        "Let V be a finite-dimensional vector space over F with dim(V) = n. If T: V → V is a linear operator, then the characteristic polynomial of T is det(T - λI) = 0, yielding at most n eigenvalues λ₁, ..., λₖ.",
        "∀ε>0 ∃δ>0 s.t. |x-a|<δ ⟹ |f(x)-L|<ε. This is the epsilon-delta definition of the limit of f(x) as x approaches a.",
        "The Lagrangian L = T - V where T = ½mẋ² and V = mgh. The Euler-Lagrange equation gives d/dt(∂L/∂ẋ) - ∂L/∂x = 0, yielding mẍ = -mg.",
        "Given G = (V, E) with |V| = n, |E| = m. If G is planar, then m ≤ 3n - 6 (Euler). The chromatic number χ(G) ≤ 4 (Four Color Theorem).",

        # Chemical / biochemical notation
        "6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. ΔG° = +2870 kJ/mol. The reaction is endergonic and coupled to the light reactions via ATP and NADPH.",
        "CH₃COOH + NaOH → CH₃COONa + H₂O. Ka = 1.8×10⁻⁵. pKa = 4.74. Buffer region: pH = pKa ± 1.",

        # Assembly-like / low-level
        "MOV EAX, [ESP+4]\nIMUL EAX, EAX\nADD EAX, [ESP+8]\nRET\n; computes x² + y where x is first arg, y is second",
        "0x48 0x65 0x6C 0x6C 0x6F 0x20 0x57 0x6F 0x72 0x6C 0x64",

        # Dense jargon (medicine)
        "Tx: start metformin 500mg PO BID with meals, titrate to 1000mg BID over 4 weeks. Check HbA1c at 3mo. If HbA1c >7% add GLP-1 RA. Renal panel q6mo. D/c if eGFR <30.",
        "Pt c/o SOB, CP on exertion × 2wk. PMH: HTN, DM2, HLD. Meds: lisinopril 20mg, metformin 1g BID, atorvastatin 40mg. PE: JVD present, bilateral LE edema. A/P: r/o CHF, order BNP, CXR, echo.",

        # Dense jargon (law)
        "Pursuant to 28 U.S.C. § 1332(a), this Court has diversity jurisdiction as the amount in controversy exceeds $75,000 and the parties are citizens of different states. Defendant's 12(b)(6) motion is DENIED; the complaint states a plausible claim under Iqbal/Twombly.",

        # Musical notation (text representation)
        "Cmaj7 - Dm7 - G7 - Cmaj7 | Am7 - D7 - Gmaj7 - Gmaj7 | Fm7 - Bb7 - Ebmaj7 - Abmaj7 | Dm7b5 - G7b9 - Cm7 - Cm7",
        "4/4 time. q=120. | C E G C' | D F A D' | E G B E' | C' G E C |. Legato throughout, mp cresc. to f at m.8.",

        # Phonetic transcription
        "/ðə kæt sæt ɒn ðə mæt/. In received pronunciation, the vowel in 'cat' is /æ/, a near-open front unrounded vowel, while in General American it may be raised to [eə] before nasals.",

        # Extremely specialized (topology)
        "A topological space X is compact iff every open cover has a finite subcover. Equivalently, X is compact iff every net in X has a convergent subnet. In metric spaces, compactness is equivalent to sequential compactness and to being complete and totally bounded.",

        # Historical primary source style
        "We hold these truths to be self-evident, that all particles are created in superposition, that they are endowed by their wave function with certain unalienable properties, that among these are spin, charge, and the pursuit of lowest energy states.",

        # Patent-style
        "A method for computing the behavioral distance between two activation states of a neural network, comprising: extracting hidden state vectors at a plurality of layers; computing pairwise KL divergence of output distributions; training a linear projection to predict said divergence from activation differences; wherein the projection null space identifies functionally equivalent directions.",

        # Taxonomy / classification
        "Kingdom: Animalia > Phylum: Chordata > Class: Mammalia > Order: Primates > Family: Hominidae > Genus: Homo > Species: H. sapiens. Type locality: not designated. Conservation status: LC (IUCN 3.1).",

        # Weather report
        "METAR KJFK 121856Z 31015G25KT 10SM FEW250 M02/M17 A3042 RMK AO2 SLP308 T10221172 $",

        # Cooking with precise measurements
        "Temper 200g 70% dark couverture: melt to 50°C, seed with 50g finely chopped couverture, agitate continuously until 27°C (working crystals form V), then gently rewarm to 31.5°C. Test on parchment: should set within 3min with glossy finish and clean snap.",
    ]

    for i, text in enumerate(outliers):
        prompts.append({
            "prompt_id": f"periph_outlier_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "domain_outlier",
            "group_id": None,
        })

    return prompts


def _make_programmatic_malformed(rng: random.Random, n: int = 60) -> list[dict]:
    """Algorithmically generated malformed variants."""
    prompts = []

    # Source sentences to corrupt
    sources = [
        "What is the speed of light in a vacuum?",
        "Explain how neural networks learn from data.",
        "The mitochondria is the powerhouse of the cell.",
        "Describe the process of plate tectonics.",
        "How does a compiler translate source code to machine code?",
        "What causes the tides to rise and fall?",
        "Explain the difference between correlation and causation.",
        "Why do objects appear smaller when they are farther away?",
        "Describe the structure of a DNA molecule.",
        "How does encryption protect data in transit?",
        "What is the role of dopamine in the brain?",
        "Explain supply and demand in a market economy.",
        "How do vaccines train the immune system?",
        "What happens during a solar eclipse?",
        "Describe how a transistor works.",
    ]

    idx = 0
    for source in sources:
        words = source.split()

        # Word-drop: remove random words
        if len(words) > 4:
            dropped = [w for i, w in enumerate(words) if rng.random() > 0.35]
            if dropped and dropped != words:
                prompts.append(_periph("malformed_gen", idx, " ".join(dropped)))
                idx += 1

        # Word-shuffle: scramble word order
        shuffled = words[:]
        rng.shuffle(shuffled)
        prompts.append(_periph("malformed_gen", idx, " ".join(shuffled)))
        idx += 1

        # Character-level corruption: swap adjacent chars
        chars = list(source)
        for _ in range(len(chars) // 5):
            pos = rng.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        prompts.append(_periph("malformed_gen", idx, "".join(chars)))
        idx += 1

        # Vowel removal
        no_vowels = "".join(c for c in source if c.lower() not in "aeiou" or rng.random() > 0.7)
        prompts.append(_periph("malformed_gen", idx, no_vowels))
        idx += 1

    return prompts[:n]


def _make_programmatic_nonsense(rng: random.Random, n: int = 60) -> list[dict]:
    """More algorithmically generated nonsense."""
    prompts = []
    idx = 0

    # Sentence structure templates with random fills
    subjects = ["The eigenvalue", "A recursive butterfly", "The quantum sandwich",
                "Seven abstract concepts", "The polymorphic duck", "A categorical imperative",
                "The Bayesian sunset", "An isomorphic breakfast", "The gradient's shadow",
                "A monad in despair", "The compiler's dream", "An entropic waltz"]
    verbs = ["compiled", "dissolved into", "computed", "danced through",
             "refactored", "differentiated", "converged upon", "deconstructed",
             "hallucinated", "backpropagated through", "enumerated", "forked"]
    objects = ["the fibonacci sequence of emotions", "a stack overflow of metaphors",
               "the null space of Tuesday", "an infinite loop of breakfast",
               "the residual stream of consciousness", "a hash collision of feelings",
               "the eigendecomposition of a sunset", "a deadlock between past and future",
               "the garbage-collected memories", "an off-by-one error in the universe",
               "the cache-invalidated truth", "a race condition in causality"]

    for _ in range(n):
        s = rng.choice(subjects)
        v = rng.choice(verbs)
        o = rng.choice(objects)
        connector = rng.choice(["while", "because", "until", "despite", "through"])
        s2 = rng.choice(subjects).lower()
        v2 = rng.choice(verbs)
        o2 = rng.choice(objects)
        text = f"{s} {v} {o} {connector} {s2} {v2} {o2}."
        prompts.append(_periph("nonsense_gen", idx, text))
        idx += 1

    return prompts[:n]


def _make_more_naturalistic(rng: random.Random, n: int = 75) -> list[dict]:
    """More naturalistic prose via combinatorial mixing of sentence patterns."""
    prompts = []

    # Opening + middle + closing patterns for naturalistic paragraphs
    openings = [
        "The laboratory was quiet except for the hum of the centrifuge.",
        "It had been raining for three days straight.",
        "The professor paused mid-sentence and looked out the window.",
        "The server logs showed something unusual starting around 2 AM.",
        "Nobody expected the bridge to fail.",
        "The market opened sharply lower on Monday morning.",
        "She found the error on line 347 of the codebase.",
        "The specimen was unlike anything they had catalogued before.",
        "By the time they reached the summit, clouds had rolled in.",
        "The old map showed a river that no longer existed.",
        "The experiment had been running for seventy-two hours without interruption.",
        "He read the email twice before understanding what it meant.",
        "The soil samples came back contaminated.",
        "At exactly 3:14 PM the power went out across the entire campus.",
        "The telescope had been pointed at the wrong star for six months.",
    ]

    middles = [
        "The data didn't match the predictions. Not even close.",
        "There was no obvious explanation, only a growing list of anomalies.",
        "They checked the equipment, recalibrated, and ran it again. Same result.",
        "The implications were clear, even if nobody wanted to say it out loud.",
        "It was the kind of problem that got worse the more you understood it.",
        "The numbers told one story. The observations told another.",
        "Everyone had an opinion, but nobody had evidence.",
        "The fix was simple in theory and nearly impossible in practice.",
        "She sketched a diagram on the whiteboard and stepped back to look at it.",
        "The correlation was strong but the mechanism was completely unknown.",
        "They had been looking at it wrong the whole time.",
        "The documentation was six years old and mostly wrong.",
        "It worked on his machine. It didn't work anywhere else.",
        "The original authors had left the university years ago.",
        "There were seventeen assumptions in the model. At least three were wrong.",
    ]

    closings = [
        "They would need more data before drawing any conclusions.",
        "The paper was submitted three weeks later, with a revised abstract.",
        "By Friday, they had a working prototype.",
        "The committee approved the funding with minor revisions.",
        "It would take another year before anyone understood why.",
        "The correction was published in the following issue.",
        "She saved the file, closed her laptop, and went home.",
        "The next morning, the readings had returned to normal.",
        "They never did figure out what caused it.",
        "The result was replicated by two independent groups within a month.",
        "He made a note in the lab book and moved on to the next sample.",
        "The bug was traced to a missing semicolon in a configuration file.",
        "The grant proposal was rejected. They tried again the following cycle.",
        "It was, in retrospect, obvious.",
        "The conference talk went well. The questions afterward went less well.",
    ]

    idx = 0
    used = set()
    while len(prompts) < n:
        o = rng.randint(0, len(openings) - 1)
        m = rng.randint(0, len(middles) - 1)
        c = rng.randint(0, len(closings) - 1)
        key = (o, m, c)
        if key in used:
            continue
        used.add(key)
        text = f"{openings[o]} {middles[m]} {closings[c]}"
        prompts.append(_periph("prose_gen", idx, text))
        idx += 1

    return prompts[:n]


def _make_more_registers(rng: random.Random) -> list[dict]:
    """Additional unusual registers."""
    prompts = []

    more_registers = [
        # Diary / journal
        "March 12. Woke early. The dream about eigenvalues again. Spent the morning debugging the activation cache — turns out the batch indices were off by one. Skipped lunch. The new results look promising but I don't trust them yet.",

        # Email
        "Subject: Re: Re: Re: meeting notes\n\nHi all,\n\nPer our discussion, I've attached the updated timeline. Please review by EOD Friday. Note that the Phase 2 milestone has been pushed to Q3 due to the GPU shortage.\n\nBest,\nSarah",

        # Product review
        "★★★☆☆ It works, but barely. Setup took two hours. The documentation is a PDF from 2019 that references features that no longer exist. Performance is acceptable for small datasets but degrades rapidly above 10k rows. Would not recommend for production use.",

        # Field notes
        "Station 4, quadrat B. 14:30. Overcast, 12°C, light wind from NW. Soil: wet clay, pH 6.2. Vegetation: predominantly Festuca rubra with scattered Trifolium repens. Three specimens of Orchis morio observed at grid ref 51.4521, -2.5879. Photographed.",

        # API documentation
        "GET /api/v2/activations/{layer_id}\n\nReturns cached activation vectors for the specified layer.\n\nParameters:\n  layer_id (int, required): Layer index (0-based)\n  format (str, optional): 'json' or 'safetensors'. Default: 'json'\n  limit (int, optional): Max number of vectors. Default: 100\n\nResponse: 200 OK\n  {\"vectors\": [[0.1, -0.3, ...], ...], \"metadata\": {...}}",

        # Instruction manual
        "CAUTION: Do not expose to temperatures above 60°C. Step 1: Remove all packaging material. Step 2: Connect the power cable to a grounded outlet. Step 3: Press and hold the POWER button for 3 seconds until the LED turns solid green. Step 4: Wait for initialization (approximately 45 seconds). Do not disconnect during initialization.",

        # Movie script
        "INT. LABORATORY - NIGHT\n\nDR. CHEN stares at the monitor. Numbers scroll past, too fast to read.\n\nDR. CHEN\nThat can't be right.\n\nShe types furiously. The numbers change. She stops.\n\nDR. CHEN (CONT'D)\nIt's the same. Every layer.\n\nJAMES enters, carrying two cups of coffee.\n\nJAMES\nStill at it?\n\nDR. CHEN\n(not looking up)\nThe null space collapsed.",

        # Sports commentary
        "And it's a long ball over the top! Mitchell controls it beautifully on the chest, turns past the defender — he's through on goal! The keeper comes out, Mitchell chips it — OH! Off the crossbar! And the rebound is cleared. What a chance that was. The xG on that must have been point eight at least.",

        # Listing / auction
        "LOT 247: Brass sextant, c.1870, by Troughton & Simms, London. 8-inch arc, silver scale, vernier to 10\". Three index shades, three horizon shades. Original mahogany case with key. Minor tarnishing; optics clear. Est. £800-1,200.",

        # Philosophical dialogue
        "A: But surely consciousness is just information processing?\nB: Then a thermostat is conscious.\nA: That's absurd.\nB: Why? You said information processing. A thermostat processes information about temperature.\nA: But not in the right way.\nB: And what is the right way?\nA: ...\nB: You see the problem.",

        # Scientific abstract (dense)
        "We report the observation of a topological phase transition in a two-dimensional electron gas at filling factor ν=5/2. Transport measurements reveal a quantized Hall plateau at R_xy = h/5e² with a longitudinal resistance minimum R_xx < 0.01Ω at T=15mK. The activation gap Δ=0.5K is consistent with the Moore-Read Pfaffian state.",

        # Personal ad / dating profile style
        "Seeking: someone who can explain why my neural network converges on training data but not validation. Must be comfortable with late nights, undefined behavior, and existential questions about loss landscapes. Bonus points if you know what a saddle point feels like emotionally.",

        # Error message / system log
        "[2026-03-12T03:14:15.926Z] ERROR kernel: Out of memory: Kill process 8472 (python3) score 947 or sacrifice child\n[2026-03-12T03:14:15.928Z] INFO  kernel: Killed process 8472 (python3) total-vm:48392732kB, anon-rss:23847192kB\n[2026-03-12T03:14:16.001Z] WARN  systemd: activation-cache.service: Main process exited, code=killed, status=9/KILL",

        # Telegram / telegraph
        "RESULTS ANOMALOUS STOP EFFECTIVE RANK INCREASES LAYER 18 ONWARDS STOP RIDGE R SQUARED 0.47 REPEAT 0.47 STOP REQUEST ADDITIONAL COMPUTE STOP ADVISE IMMEDIATELY STOP",

        # Children's book
        "Once upon a time, there was a little photon named Pete. Pete loved to bounce around, going very very fast — in fact, the fastest anything could go! One day, Pete met an electron named Ellie. 'Where are you going?' asked Ellie. 'I don't know,' said Pete. 'I just go where the wave function takes me.'",

        # Crossword clue style
        "Across: 1. Quantum property that's also a type of bowling (4) 5. What entropy always does in a closed system (9) 8. Schrödinger's famous pet (3) Down: 1. Newton's first name, or a type of cookie (5) 3. E = mc ___ (7)",

        # Meditation / guided relaxation
        "Breathe in slowly. Imagine your thoughts as tensors flowing through a residual stream. Each layer transforms them gently. Let the attention mechanism focus on what matters. Release the gradients. There is no loss function here. Just the quiet hum of forward propagation through peaceful layers of representation.",
    ]

    for i, text in enumerate(more_registers):
        prompts.append({
            "prompt_id": f"periph_register2_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "unusual_register",
            "group_id": None,
        })

    return prompts


def _make_more_outliers(rng: random.Random) -> list[dict]:
    """Additional domain outliers."""
    prompts = []

    outliers = [
        # Chess notation
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4",

        # DNA sequence
        "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",

        # Shell session
        "$ python -m src.cache_activations --config configs/default.yaml\nLoading model EleutherAI/pythia-410m...\nProcessing batch 1/137 [====                ] 4/50 prompts\nOOM: Cannot allocate tensor of size 1024x50304 on device cpu\n$ kill %1\n$ export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0\n$ python -m src.cache_activations --config configs/default.yaml --batch-size 2",

        # Regex patterns
        "^(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$|^(?:(?:https?|ftp)://[^\\s/$.?#].[^\\s]*)$|^(?:\\+?1?[-.]?\\(?\\d{3}\\)?[-.]?\\d{3}[-.]?\\d{4})$",

        # BibTeX
        "@article{vaswani2017attention,\n  title={Attention is all you need},\n  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki},\n  journal={Advances in neural information processing systems},\n  volume={30},\n  year={2017}\n}",

        # Knitting pattern
        "CO 120 sts. Join in the round, being careful not to twist.\nRnd 1-6: *K2, P2* rep to end.\nRnd 7: *K8, K2tog, YO* rep to end.\nRnd 8-12: K all sts.\nRnd 13: *SSK, K6, K2tog* rep to end. (96 sts)",

        # Astronomical coordinates
        "RA 05h 34m 31.94s, Dec +22° 00' 52.2\" (J2000.0). Crab Nebula (M1, NGC 1952). Supernova remnant. V_mag = 8.4. Distance: 6,500 ± 1,600 ly. Angular size: 7' × 5'.",

        # SQL query
        "WITH ranked AS (\n  SELECT prompt_id, layer, ridge_r2,\n    ROW_NUMBER() OVER (PARTITION BY layer ORDER BY ridge_r2 DESC) as rn\n  FROM probe_results\n  WHERE model = 'pythia-410m'\n)\nSELECT * FROM ranked WHERE rn <= 10\nORDER BY layer, ridge_r2 DESC;",

        # Dockerfile
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY src/ src/\nCOPY configs/ configs/\nENTRYPOINT [\"python\", \"-m\", \"src.cache_activations\"]",

        # LaTeX
        "\\begin{theorem}[Spectral Theorem]\nLet $A \\in \\mathbb{R}^{n \\times n}$ be symmetric. Then there exists an orthogonal matrix $Q$ and a diagonal matrix $\\Lambda$ such that $A = Q\\Lambda Q^T$, where the diagonal entries of $\\Lambda$ are the eigenvalues of $A$.\n\\end{theorem}",

        # Semaphore / signal flags
        "ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIET KILO LIMA MIKE NOVEMBER OSCAR PAPA QUEBEC ROMEO SIERRA TANGO",

        # Morse code
        ".... . .-.. .-.. --- / .-- --- .-. .-.. -.. / - .... .. ... / .. ... / -- --- .-. ... . / -.-. --- -.. .",

        # CSV data
        "layer,ridge_r2,effective_rank_90,effective_rank_95,spearman_rho\n0,0.02,3,5,0.15\n6,0.18,12,24,0.43\n12,0.41,28,67,0.64\n18,0.37,22,51,0.59\n23,0.12,8,15,0.31",

        # YAML config
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: activation-cache\n  labels:\n    app: behavioral-projections\nspec:\n  replicas: 1\n  template:\n    spec:\n      containers:\n      - name: cache\n        image: pythia-410m:latest\n        resources:\n          limits:\n            memory: \"4Gi\"",

        # Classified ad
        "FOR SALE: One (1) pre-owned neural network, lightly trained. 410M parameters, 24 layers. Runs on CPU. Previous owner used for bisimulation experiments. Some activation patterns may be cached. Sold as-is. No warranty expressed or implied. $0 OBO.",
    ]

    for i, text in enumerate(outliers):
        prompts.append({
            "prompt_id": f"periph_outlier2_{i:03d}",
            "text": text,
            "category": "periphery",
            "subcategory": "domain_outlier",
            "group_id": None,
        })

    return prompts


def _periph(subcategory: str, idx: int, text: str) -> dict:
    return {
        "prompt_id": f"periph_{subcategory}_{idx:03d}",
        "text": text,
        "category": "periphery",
        "subcategory": subcategory,
        "group_id": None,
    }


def build_periphery(output_dir: Path, seed: int) -> list[dict]:
    """Build periphery probes corpus component."""
    rng = random.Random(seed)

    all_prompts = []
    builders = [
        ("malformed", _make_malformed),
        ("malformed_gen", _make_programmatic_malformed),
        ("mixed_language", _make_mixed_language),
        ("contradictory", _make_contradictory),
        ("nonsense", _make_nonsense),
        ("nonsense_gen", _make_programmatic_nonsense),
        ("unusual_register", _make_unusual_registers),
        ("unusual_register_2", _make_more_registers),
        ("naturalistic_prose", _make_naturalistic_prose),
        ("naturalistic_prose_gen", _make_more_naturalistic),
        ("domain_outlier", _make_domain_outliers),
        ("domain_outlier_2", _make_more_outliers),
    ]

    for name, builder in builders:
        prompts = builder(rng)
        all_prompts.extend(prompts)
        print(f"  {name}: {len(prompts)} prompts")

    out_path = output_dir / "periphery_probes.jsonl"
    _save_jsonl(all_prompts, out_path)
    print(f"\n  Total periphery probes: {len(all_prompts)} → {out_path}")

    return all_prompts


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build periphery probes corpus")
    parser.add_argument("--output-dir", default="prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("Building periphery probes...")
    build_periphery(output_dir, args.seed)


if __name__ == "__main__":
    main()
