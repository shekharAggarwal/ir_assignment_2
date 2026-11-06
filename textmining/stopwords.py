"""Stop-word lists (embedded so the app works offline, e.g. on a machine without NLTK data)."""

# Standard English stop words (same coverage as the classic NLTK / SMART short lists).
ENGLISH_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just let me more most my myself no nor not now of off
on once only or other ought our ours ourselves out over own same shall she should so some such than that the
their theirs them themselves then there these they this those through to too under until up us very was we were
what when where which while who whom why will with would you your yours yourself yourselves
s t d ll m o re ve y ain ma
""".split())

# Negation words carry sentiment in reviews ("not working", "no sound"), so the pipeline can keep them.
NEGATIONS = frozenset({"no", "not", "nor", "never", "none", "nothing", "neither", "nobody", "nowhere",
                       "cannot", "without"})

# Review boilerplate that appears in every product category and adds no topical signal.
REVIEW_STOPWORDS = ("product", "products", "item", "items", "amazon", "buy", "bought", "purchase", "purchased",
                    "would", "could", "also", "get", "got", "one", "really")
