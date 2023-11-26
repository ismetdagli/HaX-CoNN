def contains_blacklisted_term(layer_name):
    blacklist = ["reformatter"]  # , "input", "output"]
    return any(blacklisted in layer_name for blacklisted in blacklist)


def split_layer_names(layer_name):
    return sum(
        (comp.split("||") for comp in layer_name.replace(" ", "").split("+")), []
    )


def filter_real_layers(layer_components, real_layers):
    """Filters out the real layers from the layer components."""
    filtered_layers = []
    for comp in layer_components:
        if comp in real_layers and 'reformatter' not in comp.lower():
            filtered_layers.append(comp)
        else:
            print(f"Filtered out layer: {comp}")  # Debugging print
    return filtered_layers

def isLayerReal(layer_components, real_layers):
    for comp in layer_components:
        if 'reformatter' in comp.lower():
            return False
    return True

def count_unique_layers(layer_components):
    return len(set(layer_components))


def parse_layer_info(entry, real_layers):
    if (
        "name" in entry
        and "averageMs" in entry
        and not contains_blacklisted_term(entry["name"])
    ):
        layer_components = split_layer_names(entry["name"])
        filtered_layers = filter_real_layers(layer_components, real_layers)
        layer_count = count_unique_layers(filtered_layers)

        return {
            "name": entry["name"],
            "average_time_ms": entry.get("averageMs", 0),
            "layer_count": layer_count,
        }
    return None
