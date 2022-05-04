import re


class StringInputParser:
    def __init__(self, input_string: str, **kwargs):
        self.input_string = input_string
        self.kwargs = kwargs
        self._known_elements = ["x"]
        self._library_elements = ["near", "Constant", "Expression"]
        self._library = "FEN"
        self._splitting_elements = [r"\*?\*", r"\+", r"-", r"/", r"\(", r"\)", r","]
        self._test_elements()
        self._test_kwargs()

    def _test_kwargs(self):
        failures = {}
        for key, value in self.kwargs.items():
            try:
                float(value)
            except:
                failures[key] = value
        if failures:
            raise ValueError(f"Got an unexpected kwarg(s) '{failures}'")

    def _split_input(self):
        scientific_notation_re = r"[0-9]*\.[0-9]+[eE][+-]?[0-9]+|[0-9]+[eE][+-]?[0-9]+"
        return [
            r.strip()
            for r in re.split(
                "|".join(self._splitting_elements),
                "".join(re.split(scientific_notation_re, self.input_string)),
            )
            if len(r.strip()) > 0
        ]

    def _test_elements(self):
        failures = []
        for e in self._split_input():
            if e in self._known_elements:
                continue
            elif e in self._library_elements:
                e = "FEN." + e
            elif e[0] == "x":
                self._check_x_dimension(e)
                continue
            elif e in self.kwargs.keys():
                self._check_kwargs_vals(self.kwargs[e])
                continue
            elif e.isnumeric():
                continue
            else:
                try:
                    float(e)
                    continue
                except:
                    failures.append(e)
        if failures:
            raise ValueError(f"Got an unexpected symbol(s) '{failures}'")

    def _check_x_dimension(self, x):
        # How do we know dimension? At least we can double check it's an integer
        return

    def _check_kwargs_vals(self, v):
        # I think we pretty much only want to allow natural numbers?
        return