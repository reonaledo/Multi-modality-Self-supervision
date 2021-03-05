from models.mmbt import MultimodalBertClf


MODELS = {
    "mmbt": MultimodalBertClf,
}


def get_model(args):
    return MODELS[args.model](args)
