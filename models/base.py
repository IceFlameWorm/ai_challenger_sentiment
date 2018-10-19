class BaseModel(object):
    def fit(self, inputs, outputs, *args, **kwargs):
        raise NotImplementedError

    def fit_gen(self, inp_gen, *args, **kwargs):
        raise NotImplementedError

    def predict(self, inputs, *args, **kwargs):
        raise NotImplementedError


class SingleModel(BaseModel):
    def __init__(self,
                 model_file = None,
                 weights_file = None,
                 *args,
                 **kwargs):
        if model_file is not None:
            self._model = self._load(model_file, weights_file)
        else:
            self._model = self._build(*args, **kwargs)
            if weights_file is not None:
                self.load_weights(weights_file)

    def _build(self, *args, **kwargs):
        raise NotImplementedError

    def _load(self, model_file, weights_file):
        raise NotImplementedError

    def load_weights(self, weigths_file):
        raise NotImplementedError

    def save(self, model_file, weights_file):
        raise NotImplementedError


class CompositeModel(BaseModel):
    def __init__(self, components, *args, **kwargs):
        self._comps = self._instantiate_components(components)

    def _instantiate_components(self, components):
        res = []
        for item in components:
            if isinstance(item, SingleModel):
                res.append(item)
            elif isinstance(item, tuple):
                Model, kwargs = item
                res.append(Model(**kwargs))
        return res

    def fit(self, inputs, outputs, *args, **kwargs):
        for i, comp in enumerate(self._comps):
            comp.fit(inputs[i], outputs[i], *args, **kwargs)

    def predict(self, inputs, *args, **kwargs):
        res = []
        for comp in self._comps:
            res.append(comp.predict(inputs, *args, **kwargs))
        return res

    def load_weights(self, weights_files):
        for comp, wf in zip(self._comps, weights_files):
            comp.load_weights(wf)


