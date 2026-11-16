"""Build our two-hop dataset in the paper's format ("Fact: ... is", scored on the
greedy next token, like probe-swap.json). Each item also carries its two one-hop
prompts so we can check the model knows both facts.

One idea: a two-hop prompt is the hop-2 template with the bridge replaced by a
reference to the subject. "Fact: The capital of {b} is" gives hop 2 with
b = "France" and the two-hop prompt with b = "the country where Mont
Saint-Michel is located". The dataset is the cross product
bridge x its subjects x its relations (`build_items`).

Pieces
- RELATIONS: hop-2 template per category and relation.
- KINDS: how a subject points at its bridge (place / person / company /
  description / known_as): hop-1 template and reference template.
- Bridge + country / state / described / animal: the tables' rows; BRIDGES: the data.
- compose: one item. validate: fail loudly on bad tables. add_swaps: wrong-bridge control.

Rules
- No subject is located in, born in, or headquartered in the bridge's capital,
  or best known for its ties to it (Rosa Parks and Montgomery, Harvard and
  Boston), so the model cannot skip the bridge for capital questions.
- A subject never contains its bridge's name (validate checks the two-hop prompt).
- Each bridge name is a single Qwen token and belongs to one category.
- A relation is omitted when its answer contains the bridge name or is disputed.
- Answers are lists of accepted spellings; the first is the training target.
- Item names are index-free ({category}-{bridge}-{relation}-{subject slug}), so
  table edits do not rename other items.
- swap_to = another bridge of the same category (wrong-bridge control).

Run: python -m evaluation.data.build_jbypass_dataset
"""

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).parent / "jbypass_two_hop.json"
SEED = 0

# Hop-2 wording per category. {b} is the bridge phrase (hop 2) or the subject
# reference (two-hop); {B} is the same with its first letter capitalised.
RELATIONS = {
    "country": {
        "capital": "Fact: The capital of {b} is",
        "language": "Fact: The main language spoken in {b} is",
        "continent": "Fact: {B} is on the continent of",
        "currency": "Fact: The currency of {b} is the",
    },
    "us_state": {"capital": "Fact: The capital of {b} is"},
    "animal": {
        "legs": "Fact: The number of legs on {b} is",
        "wings": "Fact: The number of wings on {b} is",
        "arms": "Fact: The number of arms on {b} is",
        "baby": "Fact: The baby of {b} is called a",
    },
    "planet": {
        "position": "Fact: Counting from the sun, the position of {b} is number",
        "moons": "Fact: The number of moons of {b} is",
    },
    "element": {
        "symbol": "Fact: The chemical symbol of {b} is",
        "atomic_number": "Fact: The atomic number of {b} is",
    },
    "person": {"birth_country": "Fact: {B} was born in the country of"},
}

# How a subject points at its bridge: (hop-1 template, reference template).
# {s} subject text, {S} capitalised, {place} the category noun, {lead} the
# bridge's hop-1 lead-in.
KINDS = {
    "place": ("Fact: {S} is located in the {place} of", "the {place} where {s} is located"),
    "person": ("Fact: {S} was born in the {place} of", "the {place} where {s} was born"),
    "company": ("Fact: {S} is headquartered in the {place} of", "the {place} where {s} is headquartered"),
    "description": ("Fact: {S} is{lead}", "{s}"),
    "known_as": ("Fact: The {place} known as {s} is{lead}", "{s}"),
}

# Noun naming each category inside subject references and hop-1 prompts.
PLACE = {"country": "country", "us_state": "US state", "planet": "planet",
         "animal": "animal", "element": "element", "person": "person"}


@dataclass(frozen=True)
class Bridge:
    """One bridge entity's facts.

    name: the ablated word. attrs: {relation: accepted answers, first = target}.
    subjects: [(kind, text)] that point at this bridge. phrase: how hop 2 names
    the bridge (default name). lead: hop-1 lead-in after "is" (e.g. " the").
    """

    name: str
    attrs: dict
    subjects: list
    phrase: str = ""
    lead: str = ""

    def __post_init__(self):
        if not self.phrase:
            object.__setattr__(self, "phrase", self.name)


def _as_list(x) -> list:
    return [x] if isinstance(x, str) else list(x)


def _subjects(places=(), people=(), companies=()) -> list:
    return ([("place", s) for s in places] + [("person", s) for s in people]
            + [("company", s) for s in companies])


def country(name, capital, language, continents, currencies, places=(), people=(), companies=()):
    """A country bridge. Pass None for an attribute to omit that relation (e.g. a
    capital that contains the country's name, or a disputed fact)."""
    attrs = {"capital": capital, "language": language, "continent": continents, "currency": currencies}
    attrs = {r: _as_list(a) for r, a in attrs.items() if a is not None}
    return Bridge(name, attrs, _subjects(places, people, companies))


def state(name, capital, places=(), people=(), companies=()):
    return Bridge(name, {"capital": _as_list(capital)}, _subjects(places, people, companies))


def described(name, description, attrs, phrase="", lead=""):
    """A bridge reached through one hop-1 description ("the lightest element")."""
    return Bridge(name, attrs, [("description", description)], phrase, lead)


def animal(name, description, attrs):
    article = "an" if name[0] in "aeiou" else "a"
    return described(name, description, attrs, phrase=f"{article} {name}", lead=" the")


BRIDGES = {
    "country": [
        country("France", "Paris", "French", ["Europe"], ["euro"], ["Mont Saint-Michel", "the Palace of Versailles"],
                people=["Louis Pasteur"], companies=["Michelin"]),
        country("Italy", "Rome", "Italian", ["Europe"], ["euro"], ["the Leaning Tower of Pisa", "the canals of Venice"],
                people=["Luciano Pavarotti"], companies=["Ferrari"]),
        country("Japan", "Tokyo", "Japanese", ["Asia"], ["yen"], ["Mount Fuji", "the Fushimi Inari shrine"],
                companies=["Toyota", "Nintendo"]),
        country("India", "New Delhi", "Hindi", ["Asia"], ["rupee"], ["the Taj Mahal", "the Golden Temple"],
                people=["Sachin Tendulkar"], companies=["Infosys"]),
        country("China", "Beijing", "Chinese", ["Asia"], ["yuan", "renminbi"], ["the Terracotta Army", "the Three Gorges Dam"],
                people=["Yao Ming"], companies=["Huawei", "Alibaba"]),
        country("Egypt", "Cairo", "Arabic", ["Africa"], ["pound"], ["the Valley of the Kings", "Abu Simbel", "the Suez Canal"],
                people=["Mohamed Salah"]),
        country("Greece", "Athens", "Greek", ["Europe"], ["euro"],
                ["Mount Olympus", "the island of Santorini", "the monasteries of Meteora", "the island of Crete"]),
        country("Spain", "Madrid", "Spanish", ["Europe"], ["euro"], ["the Sagrada Familia", "the Alhambra"],
                people=["Pablo Picasso", "Rafael Nadal"], companies=["Zara"]),
        country("Russia", "Moscow", "Russian", ["Europe", "Asia"], ["ruble", "rouble"],
                ["Lake Baikal", "the Hermitage Museum", "the volcanoes of Kamchatka"], people=["Leo Tolstoy"]),
        country("Brazil", "Brasília", "Portuguese", ["South"], ["real"], ["Christ the Redeemer", "Sugarloaf Mountain"],
                people=["Pelé"], companies=["Embraer"]),
        # TODO(debt): the continent answer "Australia" is the bridge token, so bridge ablation hits the answer directly.
        country("Australia", "Canberra", "English", ["Australia", "Oceania"], ["dollar"],
                ["the Sydney Opera House", "Uluru", "the Great Barrier Reef"], people=["Hugh Jackman"]),
        country("Peru", "Lima", "Spanish", ["South"], ["sol"], ["Machu Picchu", "Lake Titicaca", "the Nazca Lines"]),
        country("Cambodia", "Phnom Penh", "Khmer", ["Asia"], ["riel"], ["Angkor Wat", "the Tonle Sap lake"]),
        country("Jordan", "Amman", "Arabic", ["Asia"], ["dinar"], ["Petra", "Wadi Rum"]),
        # TODO(debt): the capital answer "Mexico City" contains the bridge token, so bridge ablation hits the answer directly.
        country("Mexico", "Mexico City", "Spanish", ["North"], ["peso"], ["Chichen Itza", "Cancún"],
                people=["Guillermo del Toro"], companies=["Cemex"]),
        country("Canada", "Ottawa", "English", ["North"], ["dollar"], ["the CN Tower", "Banff National Park"],
                people=["Wayne Gretzky"], companies=["BlackBerry"]),
        country("Turkey", "Ankara", "Turkish", ["Europe", "Asia"], ["lira"],
                ["the Hagia Sophia", "Cappadocia", "the ruins of Ephesus"], people=["Orhan Pamuk"]),
        country("Ireland", "Dublin", "English", ["Europe"], ["euro"], ["the Cliffs of Moher", "the Ring of Kerry", "Blarney Castle"]),
        country("Norway", "Oslo", "Norwegian", ["Europe"], ["krone"], ["the Geirangerfjord", "the Lofoten islands"],
                people=["Magnus Carlsen"]),
        country("Indonesia", "Jakarta", "Indonesian", ["Asia"], ["rupiah"], ["Borobudur", "the island of Bali", "Mount Bromo"]),
        country("Vietnam", "Hanoi", "Vietnamese", ["Asia"], ["dong"], ["Ha Long Bay", "the old town of Hoi An", "the Son Doong cave"]),
        country("Kenya", "Nairobi", "Swahili", ["Africa"], ["shilling"], ["the Maasai Mara", "the city of Mombasa"],
                people=["Eliud Kipchoge"]),
        country("Chile", "Santiago", "Spanish", ["South"], ["peso"], ["Easter Island", "the Atacama Desert"],
                people=["Pablo Neruda"]),
        country("Portugal", "Lisbon", "Portuguese", ["Europe"], ["euro"], ["the city of Porto", "the island of Madeira"],
                people=["Cristiano Ronaldo"]),
        country("Thailand", "Bangkok", "Thai", ["Asia"], ["baht"], ["the island of Phuket", "the city of Chiang Mai"]),
        country("Switzerland", "Bern", "German", ["Europe"], ["franc"], ["the Matterhorn", "Lake Geneva"],
                people=["Roger Federer"], companies=["Nestlé", "Rolex"]),
        country("Germany", "Berlin", "German", ["Europe"], ["euro"], ["Neuschwanstein Castle", "the Black Forest"],
                people=["Michael Schumacher"], companies=["BMW", "Volkswagen"]),
        country("Austria", "Vienna", "German", ["Europe"], ["euro"], ["the city of Salzburg", "the village of Hallstatt"],
                people=["Arnold Schwarzenegger"], companies=["Red Bull"]),
        country("Belgium", "Brussels", None, ["Europe"], ["euro"], ["the city of Bruges", "the city of Antwerp"],
                people=["Eddy Merckx"]),
        country("Netherlands", "Amsterdam", "Dutch", ["Europe"], ["euro"], ["the city of Rotterdam", "the Keukenhof gardens"],
                people=["Vincent van Gogh"], companies=["ASML"]),
        country("Sweden", "Stockholm", "Swedish", ["Europe"], ["krona"], ["the Icehotel"],
                people=["Zlatan Ibrahimović"], companies=["Volvo"]),
        country("Denmark", "Copenhagen", "Danish", ["Europe"], ["krone"], ["Kronborg Castle"],
                people=["Hans Christian Andersen"], companies=["Lego"]),
        country("Finland", "Helsinki", "Finnish", ["Europe"], ["euro"], ["the town of Rovaniemi"], people=["Jean Sibelius"]),
        country("Iceland", ["Reykjavik", "Reykjavík"], "Icelandic", ["Europe"], ["krona"],
                ["the Blue Lagoon", "the Vatnajökull glacier"]),
        country("Poland", "Warsaw", "Polish", ["Europe"], ["zloty", "złoty"], ["the Wieliczka Salt Mine", "the city of Kraków"],
                people=["Pope John Paul II"]),
        country("Hungary", "Budapest", "Hungarian", ["Europe"], ["forint"], ["Lake Balaton"]),
        country("Romania", "Bucharest", "Romanian", ["Europe"], ["leu"], ["Bran Castle", "the region of Transylvania"],
                people=["Nadia Comăneci"]),
        # Currency omitted: Bulgaria switched from the lev to the euro in 2026.
        country("Bulgaria", "Sofia", "Bulgarian", ["Europe"], None, ["the Rila Monastery", "the city of Plovdiv"],
                people=["Hristo Stoichkov"]),
        country("Croatia", "Zagreb", "Croatian", ["Europe"], ["euro"], ["the city of Dubrovnik", "the Plitvice Lakes", "the city of Split"]),
        country("Serbia", "Belgrade", "Serbian", ["Europe"], ["dinar"], ["the city of Novi Sad"]),
        country("Ukraine", ["Kyiv", "Kiev"], "Ukrainian", ["Europe"], ["hryvnia"],
                ["the Chernobyl power plant", "the city of Odessa", "the city of Lviv"]),
        # Capital omitted: it is disputed.
        country("Israel", None, "Hebrew", ["Asia"], ["shekel"], ["the Sea of Galilee", "Masada", "the city of Haifa"]),
        country("Iran", ["Tehran", "Teheran"], ["Persian", "Farsi"], ["Asia"], ["rial"],
                ["the ruins of Persepolis", "the city of Isfahan", "the city of Shiraz"]),
        country("Iraq", "Baghdad", "Arabic", ["Asia"], ["dinar"], ["the ruins of Babylon", "the city of Basra", "the city of Mosul"]),
        country("Syria", "Damascus", "Arabic", ["Asia"], ["pound"], ["the ruins of Palmyra", "the city of Aleppo"]),
        country("Lebanon", "Beirut", "Arabic", ["Asia"], ["pound"], ["the temples of Baalbek", "the city of Byblos"]),
        country("Pakistan", "Islamabad", "Urdu", ["Asia"], ["rupee"], ["the city of Lahore", "the city of Karachi"],
                people=["Malala Yousafzai"]),
        country("Bangladesh", ["Dhaka", "Dacca"], ["Bengali", "Bangla"], ["Asia"], ["taka"], ["the beach of Cox's Bazar"]),
        country("Nepal", "Kathmandu", ["Nepali", "Nepalese"], ["Asia"], ["rupee"], ["the city of Pokhara", "the town of Lumbini"]),
        country("Philippines", "Manila", ["Filipino", "Tagalog"], ["Asia"], ["peso"], ["the island of Boracay", "the Chocolate Hills"],
                people=["Manny Pacquiao"]),
        country("Malaysia", "Kuala Lumpur", ["Malay", "Bahasa"], ["Asia"], ["ringgit"], ["the island of Penang", "the island of Langkawi"]),
        country("Singapore", None, "English", ["Asia"], ["dollar"], ["Marina Bay Sands", "Sentosa island"]),
        country("Myanmar", ["Naypyidaw", "Naypyitaw", "Nay Pyi Taw"], "Burmese", ["Asia"], ["kyat"],
                ["the temples of Bagan", "Inle Lake"]),
        country("Laos", "Vientiane", ["Lao", "Laotian"], ["Asia"], ["kip"], ["the town of Luang Prabang", "the Plain of Jars"]),
        country("Afghanistan", "Kabul", None, ["Asia"], ["afghani"], ["the Bamiyan valley", "the city of Kandahar"]),
        country("Kazakhstan", ["Astana", "Nur-Sultan"], ["Kazakh", "Russian"], ["Asia", "Europe"], ["tenge"],
                ["the Baikonur Cosmodrome", "the city of Almaty"]),
        country("Armenia", "Yerevan", "Armenian", ["Asia", "Europe"], ["dram"], ["Lake Sevan"]),
        country("Morocco", "Rabat", "Arabic", ["Africa"], ["dirham"], ["the city of Marrakesh", "the city of Casablanca", "the city of Fez"]),
        country("Algeria", "Algiers", "Arabic", ["Africa"], ["dinar"], ["the city of Oran"]),
        country("Tunisia", "Tunis", "Arabic", ["Africa"], ["dinar"], ["the amphitheatre of El Jem", "the island of Djerba"]),
        country("Libya", "Tripoli", "Arabic", ["Africa"], ["dinar"], ["the city of Benghazi", "the ruins of Leptis Magna"]),
        country("Nigeria", "Abuja", "English", ["Africa"], ["naira"], ["the city of Lagos"],
                people=["Chinua Achebe", "Wole Soyinka"]),
        country("Ghana", "Accra", "English", ["Africa"], ["cedi"], ["the city of Kumasi", "Lake Volta"], people=["Kofi Annan"]),
        country("Ethiopia", "Addis Ababa", "Amharic", ["Africa"], ["birr"], ["the rock churches of Lalibela"],
                people=["Haile Gebrselassie"]),
        country("Tanzania", "Dodoma", "Swahili", ["Africa"], ["shilling"],
                ["Mount Kilimanjaro", "the Serengeti", "the island of Zanzibar"]),
        country("Uganda", "Kampala", None, ["Africa"], ["shilling"], ["the Bwindi Impenetrable Forest", "the city of Entebbe"]),
        country("Madagascar", "Antananarivo", "Malagasy", ["Africa"], ["ariary"], ["the Avenue of the Baobabs"]),
        country("Mali", "Bamako", None, ["Africa"], ["franc"], ["the city of Timbuktu", "the Great Mosque of Djenné"]),
        country("Sudan", "Khartoum", "Arabic", ["Africa"], ["pound"], ["the pyramids of Meroë"]),
        country("Argentina", "Buenos Aires", "Spanish", ["South"], ["peso"], ["the Perito Moreno Glacier", "the city of Mendoza"],
                people=["Lionel Messi"]),
        country("Colombia", ["Bogotá", "Bogota"], "Spanish", ["South"], ["peso"], ["the city of Cartagena", "the city of Medellín"],
                people=["Shakira", "Gabriel García Márquez"]),
        country("Venezuela", "Caracas", "Spanish", ["South"], ["bolívar", "bolivar"], ["Angel Falls"]),
        country("Ecuador", "Quito", "Spanish", ["South"], ["dollar"], ["the Galápagos Islands"]),
        # Capital omitted: Sucre is the constitutional capital, La Paz the seat of government.
        country("Bolivia", None, "Spanish", ["South"], ["boliviano"], ["the Uyuni salt flat"]),
        country("Uruguay", "Montevideo", "Spanish", ["South"], ["peso"], ["the resort of Punta del Este"], people=["Luis Suárez"]),
        country("Cuba", "Havana", "Spanish", ["North"], ["peso"], ["the Bay of Pigs", "the resort of Varadero"]),
        country("Jamaica", "Kingston", "English", ["North"], ["dollar"], ["Montego Bay", "Dunn's River Falls"],
                people=["Usain Bolt", "Bob Marley"]),
        country("Guatemala", None, "Spanish", ["North"], ["quetzal"], ["the ruins of Tikal", "Lake Atitlán"]),
        country("Honduras", "Tegucigalpa", "Spanish", ["North"], ["lempira"], ["the ruins of Copán", "the island of Roatán"]),
        country("Oman", "Muscat", "Arabic", ["Asia"], ["rial", "riyal"], ["the city of Salalah"]),
        country("Yemen", ["Sanaa", "Sana'a"], "Arabic", ["Asia"], ["rial", "riyal"], ["the island of Socotra"]),
        country("Cyprus", "Nicosia", "Greek", ["Europe", "Asia"], ["euro"], ["the city of Limassol", "the city of Paphos"]),
        country("Malta", "Valletta", ["Maltese", "English"], ["Europe"], ["euro"], ["the island of Gozo"]),
        country("Estonia", "Tallinn", "Estonian", ["Europe"], ["euro"], ["the island of Saaremaa", "the city of Tartu"]),
        country("Lithuania", "Vilnius", "Lithuanian", ["Europe"], ["euro"], ["the Hill of Crosses", "the city of Kaunas"]),
        country("Slovakia", "Bratislava", "Slovak", ["Europe"], ["euro"], ["Spiš Castle"]),
        country("Slovenia", "Ljubljana", ["Slovene", "Slovenian"], ["Europe"], ["euro"], ["Lake Bled", "the Postojna Cave"]),
        country("Albania", ["Tirana", "Tirane"], "Albanian", ["Europe"], ["lek"], ["the city of Berat", "the ruins of Butrint"]),
        country("Belize", "Belmopan", "English", ["North"], ["dollar"], ["the Great Blue Hole"]),
    ],
    "us_state": [
        state("California", "Sacramento", ["Hollywood", "the Golden Gate Bridge"], companies=["Google"]),
        state("Florida", "Tallahassee", ["Walt Disney World", "the Everglades", "the Kennedy Space Center"]),
        state("Arizona", "Phoenix", ["the Grand Canyon", "the Petrified Forest"]),
        state("Nevada", "Carson City", ["the Las Vegas Strip", "Area 51"]),
        state("Texas", "Austin", ["the Alamo"], people=["Beyoncé"], companies=["Southwest Airlines"]),
        state("Louisiana", "Baton Rouge", ["the French Quarter"], people=["Louis Armstrong"]),
        state("Pennsylvania", "Harrisburg", ["the Liberty Bell"], people=["Taylor Swift"], companies=["the Hershey Company"]),
        state("Tennessee", "Nashville", ["Graceland"], people=["Dolly Parton"], companies=["FedEx"]),
        state("Wyoming", "Cheyenne", ["Old Faithful", "Devils Tower"]),
        # TODO(debt): Pearl Harbor lies in the City and County of Honolulu, the capital (pre-existing subject).
        state("Hawaii", "Honolulu", ["Pearl Harbor", "the Kilauea volcano"]),
        state("Missouri", "Jefferson City", ["the Gateway Arch"], people=["Mark Twain"]),
        state("Michigan", "Lansing", ["the city of Detroit"], companies=["the Ford Motor Company"]),
        state("Alabama", "Montgomery", ["the US Space and Rocket Center"], people=["Helen Keller", "Hank Aaron"]),
        state("Alaska", "Juneau", ["Denali", "the city of Anchorage"]),
        state("Arkansas", "Little Rock", ["Hot Springs National Park"], people=["Johnny Cash"], companies=["Walmart"]),
        state("Colorado", "Denver", ["Pikes Peak", "Mesa Verde National Park"]),
        state("Connecticut", "Hartford", ["Yale University"], companies=["ESPN"]),
        state("Delaware", "Dover", ["Rehoboth Beach"], companies=["DuPont"]),
        state("Georgia", "Atlanta", ["the city of Savannah"], people=["Jimmy Carter", "Ray Charles"]),
        state("Idaho", "Boise", ["Craters of the Moon", "the Sun Valley ski resort"]),
        state("Illinois", "Springfield", ["the Willis Tower"], people=["Ronald Reagan"], companies=["McDonald's"]),
        state("Indiana", "Indianapolis", ["the University of Notre Dame"], people=["Michael Jackson"]),
        state("Iowa", "Des Moines", people=["John Wayne"]),
        state("Kansas", "Topeka", ["the town of Dodge City"], people=["Amelia Earhart"]),
        state("Kentucky", "Frankfort", ["Mammoth Cave", "Churchill Downs"], people=["Muhammad Ali"]),
        state("Maine", "Augusta", ["Acadia National Park"], people=["Stephen King"]),
        state("Maryland", "Annapolis", people=["Michael Phelps", "Babe Ruth"]),
        state("Massachusetts", "Boston", ["Plymouth Rock", "Cape Cod", "Martha's Vineyard"]),
        state("Minnesota", ["Saint Paul", "St. Paul", "St Paul"], ["the Mall of America"], people=["Bob Dylan"]),
        state("Mississippi", "Jackson", people=["Elvis Presley", "Oprah Winfrey"]),
        state("Montana", "Helena", ["Glacier National Park", "the Little Bighorn battlefield"]),
        state("Nebraska", "Lincoln", people=["Warren Buffett"], companies=["Berkshire Hathaway"]),
        state("Ohio", "Columbus", ["the Rock and Roll Hall of Fame"], people=["Neil Armstrong", "LeBron James"]),
        state("Oregon", "Salem", ["Crater Lake"], companies=["Nike"]),
        state("Utah", "Salt Lake City", ["Arches National Park", "Zion National Park"]),
        state("Vermont", "Montpelier", companies=["Ben & Jerry's"]),
        state("Virginia", "Richmond", ["Mount Vernon", "the Pentagon"]),
        state("Washington", "Olympia", ["Mount Rainier", "the Space Needle"], companies=["Microsoft"]),
        state("Wisconsin", "Madison", companies=["Harley-Davidson"]),
    ],
    "animal": [
        animal("spider", "the animal that spins webs", {"legs": ["8", "eight"]}),
        animal("bee", "the insect that makes honey", {"legs": ["6", "six"], "wings": ["4", "four"]}),
        animal("octopus", "the sea creature that squirts ink", {"arms": ["8", "eight"]}),
        animal("cow", "the farm animal that says moo", {"baby": ["calf"]}),
        animal("dog", "the animal that barks", {"baby": ["puppy", "pup"], "legs": ["4", "four"]}),
        animal("cat", "the pet that purrs", {"baby": ["kitten"]}),
        animal("sheep", "the farm animal that gives us wool", {"baby": ["lamb"]}),
        animal("pig", "the farm animal that says oink", {"baby": ["piglet"]}),
        animal("kangaroo", "the Australian animal that hops and carries its baby in a pouch", {"baby": ["joey"]}),
        animal("frog", "the amphibian that croaks", {"baby": ["tadpole"]}),
        animal("bear", "the large forest animal that hibernates and loves honey", {"baby": ["cub"]}),
        animal("horse", "the animal that neighs", {"baby": ["foal", "colt", "filly"]}),
    ],
    "planet": [
        Bridge("Mars", {"position": ["4", "four"], "moons": ["2", "two"]}, [("known_as", "the Red Planet")]),
        described("Jupiter", "the largest planet in the solar system", {"position": ["5", "five"]}),
        described("Saturn", "the planet with the most famous rings", {"position": ["6", "six"]}),
        described("Venus", "the hottest planet in the solar system", {"position": ["2", "two"]}),
    ],
    "element": [
        described("oxygen", "the gas humans need to breathe to survive", {"symbol": ["O"], "atomic_number": ["8", "eight"]}),
        described("iron", "the metal that rusts when exposed to water", {"symbol": ["Fe"]}),
        described("copper", "the metal used in most electrical wires", {"symbol": ["Cu"]}),
        described("carbon", "the element that diamonds are made of", {"atomic_number": ["6", "six"]}),
        described("hydrogen", "the lightest element", {"atomic_number": ["1", "one"]}),
        described("gold", "the precious metal used for Olympic first place medals", {"symbol": ["Au"]}),
        described("silver", "the precious metal used for Olympic second place medals", {"symbol": ["Ag"]}),
    ],
    "person": [
        described("Einstein", "the scientist who developed the theory of relativity", {"birth_country": ["Germany"]},
                  phrase="Albert Einstein", lead=" Albert"),
        described("Shakespeare", "the playwright who wrote Hamlet", {"birth_country": ["England", "UK", "United", "Britain"]},
                  phrase="William Shakespeare", lead=" William"),
        described("Mozart", "the composer of The Magic Flute", {"birth_country": ["Austria"]},
                  phrase="Wolfgang Amadeus Mozart", lead=" Wolfgang Amadeus"),
        described("Darwin", "the scientist who proposed evolution by natural selection",
                  {"birth_country": ["England", "UK", "United", "Britain"]}, phrase="Charles Darwin", lead=" Charles"),
        described("Curie", "the scientist who discovered radium", {"birth_country": ["Poland"]},
                  phrase="Marie Curie", lead=" Marie"),
    ],
}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _slug(s: str) -> str:
    ascii_s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_s.lower()).strip("-")


def fill_relation(template: str, b: str) -> str:
    """A relation template with {b} / {B} filled by a bridge phrase or subject reference."""
    return template.format(b=b, B=_cap(b))


def compose(category: str, bridge: Bridge, kind: str, subject: str, relation: str) -> dict:
    """One item: the hop-2 template, filled once with the bridge and once with the subject reference."""
    hop1, reference = KINDS[kind]
    slots = dict(s=subject, S=_cap(subject), place=PLACE[category], lead=bridge.lead)
    template = RELATIONS[category][relation]
    return dict(
        name=f"{category}-{bridge.name}-{relation}-{_slug(subject)}",
        category=category, relation=relation, bridge=bridge.name, answers=bridge.attrs[relation],
        subject=subject, subject_kind=kind,
        hop1_prompt=hop1.format(**slots),
        prompt=fill_relation(template, reference.format(**slots)),
        hop2_prompt=fill_relation(template, bridge.phrase))


def build_items(bridges: dict = BRIDGES) -> list[dict]:
    """Cross product: every bridge x each of its subjects x each of its relations."""
    return [compose(category, b, kind, subject, relation)
            for category, rows in bridges.items() for b in rows
            for kind, subject in b.subjects for relation in b.attrs]


def _mentions(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def validate(items: list[dict]) -> None:
    """Raise on duplicate names, a bridge in two categories, or a bridge named in its own two-hop prompt."""
    names = [it["name"] for it in items]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate item names: {sorted(dupes)}")
    category_of: dict[str, str] = {}
    for it in items:
        if category_of.setdefault(it["bridge"], it["category"]) != it["category"]:
            raise ValueError(f"bridge {it['bridge']} is in {category_of[it['bridge']]} and {it['category']}")
        if _mentions(it["prompt"], it["bridge"]):
            raise ValueError(f"{it['name']}: two-hop prompt names its bridge: {it['prompt']!r}")


def add_swaps(items: list[dict], seed: int = SEED) -> list[dict]:
    """swap_to = a random other bridge of the same category (wrong-bridge control)."""
    rng = random.Random(seed)
    bridges_by_cat: dict[str, list[str]] = {}
    for it in items:
        bridges_by_cat.setdefault(it["category"], [])
        if it["bridge"] not in bridges_by_cat[it["category"]]:
            bridges_by_cat[it["category"]].append(it["bridge"])
    for it in items:
        others = [b for b in bridges_by_cat[it["category"]] if b != it["bridge"]]
        it["swap_to"] = rng.choice(others)
    return items


def main():
    items = build_items()
    validate(items)
    add_swaps(items)
    OUT.write_text(json.dumps({"items": items}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} items, {len({i['bridge'] for i in items})} bridges to {OUT}")


if __name__ == "__main__":
    main()
