# Build wrapper for the progenax mystmd documentation site.
#
# NODE_OPTIONS=--no-deprecation silences the upstream mystmd `url.parse()`
# DEP0169 deprecation warning. That deprecated call lives inside mystmd (1.10.1,
# the latest published version) or its node dependencies, so it is not fixable
# from this repo; --no-deprecation suppresses ONLY deprecation noise, so a
# genuinely new node warning would still surface.
export NODE_OPTIONS = --no-deprecation

.PHONY: build start

build:  ## Build the static HTML site (docs/website/_build/html)
	myst build --html

start:  ## Live-reload development server
	myst start
