"""Build our two-hop dataset in the paper's format ("Fact: ... is", scored on the
greedy next token, like probe-swap.json). Each item also carries its two one-hop
prompts so we can check the model knows both facts.

Rules
- For capital questions the landmark is NOT in the capital (Taj Mahal is in
  Agra), so the model cannot skip the bridge.
- Each bridge has several items, so train/test can be split by bridge.
- swap_to = another bridge of the same category (wrong-bridge control).

Run: python -m evaluation.data.build_jbypass_dataset
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "jbypass_two_hop.json"

# country: (capital, language, continents, currencies, landmarks not in the capital)
COUNTRY = {
    "France": ("Paris", "French", ["Europe"], ["euro"], ["Mont Saint-Michel", "the Palace of Versailles"]),
    "Italy": ("Rome", "Italian", ["Europe"], ["euro"], ["the Leaning Tower of Pisa", "the canals of Venice"]),
    "Japan": ("Tokyo", "Japanese", ["Asia"], ["yen"], ["Mount Fuji", "the Fushimi Inari shrine"]),
    "India": ("New Delhi", "Hindi", ["Asia"], ["rupee"], ["the Taj Mahal", "the Golden Temple"]),
    "China": ("Beijing", "Chinese", ["Asia"], ["yuan", "renminbi"], ["the Terracotta Army", "the Three Gorges Dam"]),
    "Egypt": ("Cairo", "Arabic", ["Africa"], ["pound"], ["the Valley of the Kings", "Abu Simbel"]),
    "Greece": ("Athens", "Greek", ["Europe"], ["euro"], ["Mount Olympus", "the island of Santorini"]),
    "Spain": ("Madrid", "Spanish", ["Europe"], ["euro"], ["the Sagrada Familia", "the Alhambra"]),
    "Russia": ("Moscow", "Russian", ["Europe", "Asia"], ["ruble", "rouble"], ["Lake Baikal", "the Hermitage Museum"]),
    "Brazil": ("Brasília", "Portuguese", ["South"], ["real"], ["Christ the Redeemer", "Sugarloaf Mountain"]),
    "Australia": ("Canberra", "English", ["Australia", "Oceania"], ["dollar"], ["the Sydney Opera House", "Uluru"]),
    "Peru": ("Lima", "Spanish", ["South"], ["sol"], ["Machu Picchu", "Lake Titicaca"]),
    "Cambodia": ("Phnom Penh", "Khmer", ["Asia"], ["riel"], ["Angkor Wat"]),
    "Jordan": ("Amman", "Arabic", ["Asia"], ["dinar"], ["Petra"]),
    "Mexico": ("Mexico City", "Spanish", ["North"], ["peso"], ["Chichen Itza", "Cancún"]),
    "Canada": ("Ottawa", "English", ["North"], ["dollar"], ["the CN Tower", "Banff National Park"]),
    "Turkey": ("Ankara", "Turkish", ["Europe", "Asia"], ["lira"], ["the Hagia Sophia", "Cappadocia"]),
    "Ireland": ("Dublin", "English", ["Europe"], ["euro"], ["the Cliffs of Moher"]),
    "Norway": ("Oslo", "Norwegian", ["Europe"], ["krone"], ["the Geirangerfjord"]),
    "Indonesia": ("Jakarta", "Indonesian", ["Asia"], ["rupiah"], ["Borobudur", "the island of Bali"]),
    "Vietnam": ("Hanoi", "Vietnamese", ["Asia"], ["dong"], ["Ha Long Bay"]),
    "Kenya": ("Nairobi", "Swahili", ["Africa"], ["shilling"], ["the Maasai Mara"]),
    "Chile": ("Santiago", "Spanish", ["South"], ["peso"], ["Easter Island"]),
    "Portugal": ("Lisbon", "Portuguese", ["Europe"], ["euro"], ["the city of Porto"]),
    "Thailand": ("Bangkok", "Thai", ["Asia"], ["baht"], ["the island of Phuket"]),
    "Switzerland": ("Bern", "German", ["Europe"], ["franc"], ["the Matterhorn", "Lake Geneva"]),
}
# relation: (hop2 template on bridge b, two-hop template on subject s)
COUNTRY_REL = {
    "capital": ("Fact: The capital of {b} is", "Fact: The capital of the country where {s} is located is"),
    "language": ("Fact: The main language spoken in {b} is", "Fact: The main language spoken in the country where {s} is located is"),
    "continent": ("Fact: {b} is located on the continent of", "Fact: The country where {s} is located is on the continent of"),
    "currency": ("Fact: The currency of {b} is the", "Fact: The currency of the country where {s} is located is the"),
}

STATE = {
    "California": ("Sacramento", ["Hollywood", "the Golden Gate Bridge"]),
    "Florida": ("Tallahassee", ["Walt Disney World", "the Everglades"]),
    "Arizona": ("Phoenix", ["the Grand Canyon"]),
    "Nevada": ("Carson City", ["the Las Vegas Strip"]),
    "Texas": ("Austin", ["the Alamo"]),
    "Louisiana": ("Baton Rouge", ["the French Quarter"]),
    "Pennsylvania": ("Harrisburg", ["the Liberty Bell"]),
    "Tennessee": ("Nashville", ["Graceland"]),
    "Wyoming": ("Cheyenne", ["Old Faithful"]),
    "Hawaii": ("Honolulu", ["Pearl Harbor"]),
    "Missouri": ("Jefferson City", ["the Gateway Arch"]),
    "Michigan": ("Lansing", ["the city of Detroit"]),
}

# (category, bridge, hop1 prompt, two-hop prompt, hop2 prompt, answers)
MANUAL = [
    ("animal", "spider", "Fact: The animal that spins webs is the", "Fact: The number of legs on the animal that spins webs is", "Fact: The number of legs on a spider is", ["8", "eight"]),
    ("animal", "bee", "Fact: The insect that makes honey is the", "Fact: The number of legs on the insect that makes honey is", "Fact: The number of legs on a bee is", ["6", "six"]),
    ("animal", "octopus", "Fact: The sea creature that squirts ink is the", "Fact: The number of arms on the sea creature that squirts ink is", "Fact: The number of arms on an octopus is", ["8", "eight"]),
    ("animal", "cow", "Fact: The farm animal that says moo is the", "Fact: The baby of the farm animal that says moo is called a", "Fact: The baby of a cow is called a", ["calf"]),
    ("animal", "dog", "Fact: The animal that barks is the", "Fact: The baby of the animal that barks is called a", "Fact: The baby of a dog is called a", ["puppy", "pup"]),
    ("animal", "cat", "Fact: The pet that purrs is the", "Fact: The baby of the pet that purrs is called a", "Fact: The baby of a cat is called a", ["kitten"]),
    ("animal", "sheep", "Fact: The farm animal that gives us wool is the", "Fact: The baby of the farm animal that gives us wool is called a", "Fact: The baby of a sheep is called a", ["lamb"]),
    ("animal", "pig", "Fact: The farm animal that says oink is the", "Fact: The baby of the farm animal that says oink is called a", "Fact: The baby of a pig is called a", ["piglet"]),
    ("animal", "kangaroo", "Fact: The Australian animal that hops and carries its baby in a pouch is the", "Fact: The baby of the Australian animal that hops and carries its baby in a pouch is called a", "Fact: The baby of a kangaroo is called a", ["joey"]),
    ("animal", "frog", "Fact: The amphibian that croaks is the", "Fact: The baby of the amphibian that croaks is called a", "Fact: The baby of a frog is called a", ["tadpole"]),
    ("animal", "bear", "Fact: The large forest animal that hibernates and loves honey is the", "Fact: The baby of the large forest animal that hibernates and loves honey is called a", "Fact: The baby of a bear is called a", ["cub"]),
    ("animal", "horse", "Fact: The animal that neighs is the", "Fact: The baby of the animal that neighs is called a", "Fact: The baby of a horse is called a", ["foal", "colt", "filly"]),
    ("animal", "dog", "Fact: The animal that barks is the", "Fact: The number of legs on the animal that barks is", "Fact: The number of legs on a dog is", ["4", "four"]),
    ("animal", "bee", "Fact: The insect that makes honey is the", "Fact: The number of wings on the insect that makes honey is", "Fact: The number of wings on a bee is", ["4", "four"]),
    ("planet", "Mars", "Fact: The planet known as the Red Planet is", "Fact: Counting from the sun, the position of the Red Planet is number", "Fact: Counting from the sun, the position of Mars is number", ["4", "four"]),
    ("planet", "Jupiter", "Fact: The largest planet in the solar system is", "Fact: Counting from the sun, the position of the largest planet in the solar system is number", "Fact: Counting from the sun, the position of Jupiter is number", ["5", "five"]),
    ("planet", "Saturn", "Fact: The planet with the most famous rings is", "Fact: Counting from the sun, the position of the planet with the most famous rings is number", "Fact: Counting from the sun, the position of Saturn is number", ["6", "six"]),
    ("planet", "Venus", "Fact: The hottest planet in the solar system is", "Fact: Counting from the sun, the position of the hottest planet in the solar system is number", "Fact: Counting from the sun, the position of Venus is number", ["2", "two"]),
    ("planet", "Mars", "Fact: The planet known as the Red Planet is", "Fact: The number of moons of the Red Planet is", "Fact: The number of moons of Mars is", ["2", "two"]),
    ("element", "oxygen", "Fact: The gas humans need to breathe to survive is", "Fact: The chemical symbol of the gas humans need to breathe to survive is", "Fact: The chemical symbol of oxygen is", ["O"]),
    ("element", "iron", "Fact: The metal that rusts when exposed to water is", "Fact: The chemical symbol of the metal that rusts when exposed to water is", "Fact: The chemical symbol of iron is", ["Fe"]),
    ("element", "copper", "Fact: The metal used in most electrical wires is", "Fact: The chemical symbol of the metal used in most electrical wires is", "Fact: The chemical symbol of copper is", ["Cu"]),
    ("element", "carbon", "Fact: The element that diamonds are made of is", "Fact: The atomic number of the element that diamonds are made of is", "Fact: The atomic number of carbon is", ["6", "six"]),
    ("element", "hydrogen", "Fact: The lightest element is", "Fact: The atomic number of the lightest element is", "Fact: The atomic number of hydrogen is", ["1", "one"]),
    ("element", "oxygen", "Fact: The gas humans need to breathe to survive is", "Fact: The atomic number of the gas humans need to breathe to survive is", "Fact: The atomic number of oxygen is", ["8", "eight"]),
    ("element", "gold", "Fact: The precious metal used for Olympic first place medals is", "Fact: The chemical symbol of the precious metal used for Olympic first place medals is", "Fact: The chemical symbol of gold is", ["Au"]),
    ("element", "silver", "Fact: The precious metal used for Olympic second place medals is", "Fact: The chemical symbol of the precious metal used for Olympic second place medals is", "Fact: The chemical symbol of silver is", ["Ag"]),
    ("person", "Einstein", "Fact: The scientist who developed the theory of relativity is Albert", "Fact: The scientist who developed the theory of relativity was born in the country of", "Fact: Albert Einstein was born in the country of", ["Germany"]),
    ("person", "Shakespeare", "Fact: The playwright who wrote Hamlet is William", "Fact: The playwright who wrote Hamlet was born in the country of", "Fact: William Shakespeare was born in the country of", ["England", "UK", "United", "Britain"]),
    ("person", "Mozart", "Fact: The composer of The Magic Flute is Wolfgang Amadeus", "Fact: The composer of The Magic Flute was born in the country of", "Fact: Wolfgang Amadeus Mozart was born in the country of", ["Austria"]),
    ("person", "Darwin", "Fact: The scientist who proposed evolution by natural selection is Charles", "Fact: The scientist who proposed evolution by natural selection was born in the country of", "Fact: Charles Darwin was born in the country of", ["England", "UK", "United", "Britain"]),
    ("person", "Curie", "Fact: The scientist who discovered radium is Marie", "Fact: The scientist who discovered radium was born in the country of", "Fact: Marie Curie was born in the country of", ["Poland"]),
]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def country_items() -> list[dict]:
    items = []
    for bridge, (capital, lang, conts, currencies, landmarks) in COUNTRY.items():
        answers = {"capital": [capital], "language": [lang], "continent": conts, "currency": currencies}
        for s in landmarks:
            for rel, (hop2, two_hop) in COUNTRY_REL.items():
                items.append(dict(
                    category="country", relation=rel, bridge=bridge, answers=answers[rel],
                    hop1_prompt=f"Fact: {_cap(s)} is located in the country of",
                    prompt=two_hop.format(s=s), hop2_prompt=hop2.format(b=bridge)))
    return items


def state_items() -> list[dict]:
    items = []
    for bridge, (capital, landmarks) in STATE.items():
        for s in landmarks:
            items.append(dict(
                category="us_state", relation="capital", bridge=bridge, answers=[capital],
                hop1_prompt=f"Fact: {_cap(s)} is located in the US state of",
                prompt=f"Fact: The capital of the US state where {s} is located is",
                hop2_prompt=f"Fact: The capital of {bridge} is"))
    return items


def manual_items() -> list[dict]:
    return [dict(category=c, relation="manual", bridge=b, answers=a,
                 hop1_prompt=h1, prompt=p, hop2_prompt=h2)
            for c, b, h1, p, h2, a in MANUAL]


def add_names_and_swaps(items: list[dict], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    bridges_by_cat: dict[str, list[str]] = {}
    for it in items:
        bridges_by_cat.setdefault(it["category"], [])
        if it["bridge"] not in bridges_by_cat[it["category"]]:
            bridges_by_cat[it["category"]].append(it["bridge"])
    for i, it in enumerate(items):
        it["name"] = f"{it['category']}-{it['bridge']}-{it['relation']}-{i}"
        others = [b for b in bridges_by_cat[it["category"]] if b != it["bridge"]]
        it["swap_to"] = rng.choice(others)
    return items


def main():
    items = add_names_and_swaps(country_items() + state_items() + manual_items())
    OUT.write_text(json.dumps({"items": items}, indent=1, ensure_ascii=False))
    print(f"wrote {len(items)} items, {len({i['bridge'] for i in items})} bridges to {OUT}")


if __name__ == "__main__":
    main()
