"""DETR query-to-image attention inspection for the attention notebook."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch


@torch.inference_mode()
def inspect_detection(image, *, model, processor, threshold=0.7):
    """Run one image; retain original query IDs when filtering predictions."""
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    image = image.convert('RGB')
    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors='pt').to(device)
    grid = {}

    def capture_grid(module, args, features):
        # The actual final CNN feature map, not an assumed stride or square grid.
        feature, mask = features[-1]
        grid['shape'] = tuple(feature.shape[-2:])
        grid['mask'] = mask[0].detach().cpu()

    backbone = model.model.backbone
    # Transformers 5 exposes the CNN directly; older versions wrap it.
    conv_encoder = getattr(backbone, 'conv_encoder', backbone)
    hook = conv_encoder.register_forward_hook(capture_grid)
    try:
        model.eval()
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    finally:
        hook.remove()
    layers = outputs.cross_attentions
    if not layers or any(layer is None for layer in layers):
        raise RuntimeError('Load DETR with attn_implementation="eager" to get attention weights.')
    height, width = grid['shape']
    if any(layer.shape[-1] != height * width for layer in layers):
        raise RuntimeError('Cross-attention keys do not match the CNN feature grid.')

    # Match DetrImageProcessor's postprocessing, but preserve the query indices.
    # The last logit is the no-object class; class confidence is not attention.
    probabilities = outputs.logits[0].softmax(-1)
    scores, labels = probabilities[:, :-1].max(-1)
    query_ids = torch.where(scores > threshold)[0]
    cx, cy, w, h = outputs.pred_boxes[0].unbind(-1)
    boxes = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
    boxes = boxes * boxes.new_tensor([image.width, image.height, image.width, image.height])
    return {
        'image': image,
        'query_ids': query_ids.cpu(),
        'scores': scores[query_ids].cpu(),
        'labels': labels[query_ids].cpu(),
        'boxes': boxes[query_ids].cpu(),
        'id2label': model.config.id2label,
        'cross_attention': tuple(layer.detach().cpu() for layer in layers),
        'grid_shape': grid['shape'],
        'valid_mask': grid['mask'],
    }


def detection_attention_map(example, detection=0, layer=-1, head=None):
    """Return a raw H x W attention grid for a retained detection's query."""
    if not 0 <= detection < len(example['query_ids']):
        raise ValueError('Choose a detection from the retained boxes.')
    weights = example['cross_attention'][layer][0]
    query = int(example['query_ids'][detection])
    row = weights[:, query].mean(0) if head is None else weights[head, query]
    return row.reshape(example['grid_shape'])


def plot_detection_attention(example, detection=0, layer=-1, head=None, overlay=0.6):
    """Show all boxes and one query's spatial cross-attention side by side."""
    attention = detection_attention_map(example, detection, layer, head)
    image = example['image']
    query = int(example['query_ids'][detection])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    colors = plt.get_cmap('tab10')
    for ax in axes:
        ax.imshow(image)
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')

    def draw_box(ax, index, color):
        x0, y0, x1, y1 = example['boxes'][index].tolist()
        label = example['id2label'][int(example['labels'][index])]
        text = f"q{int(example['query_ids'][index])}: {label} {example['scores'][index]:.2f}"
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor=color, linewidth=2))
        ax.text(x0, y0, text, fontsize=9, color='white', va='bottom',
                bbox={'facecolor': color, 'alpha': 0.8, 'pad': 2})

    for index in range(len(example['query_ids'])):
        draw_box(axes[0], index, colors(index % 10))
    axes[0].set_title('Final predictions (q = original object-query ID)')
    # Padding keys are excluded; single-image processing normally has no padding.
    grid = np.ma.array(attention.numpy(), mask=~example['valid_mask'].numpy())
    heatmap = axes[1].imshow(
        grid, extent=(0, image.width, image.height, 0), interpolation='bilinear',
        cmap='magma', alpha=overlay, vmin=0, vmax=max(float(attention.max()), 1e-12))
    draw_box(axes[1], detection, colors(detection % 10))
    layer_index = layer % len(example['cross_attention'])
    head_label = 'mean of heads' if head is None else f'head {head}'
    axes[1].set_title(f'Query {query}: cross-attention, layer {layer_index}, {head_label}')
    fig.colorbar(heatmap, ax=axes[1], shrink=0.75, label='Attention weight per feature cell')
    return fig


def detection_attention_view(example, *, detection=0, layer=-1, head=None, overlay=0.6):
    """Start on a retained box index, layer and head; controls reuse recorded weights.

    ``detection`` is the zero-based index in the filtered boxes, not a query ID.
    ``layer=-1`` selects the last decoder layer; ``head=None`` averages heads.
    """
    import ipywidgets as widgets
    from IPython.display import display

    if len(example['query_ids']) == 0:
        print('No boxes above the threshold. Lower threshold and rerun inspect_detection.')
        return None
    layers = example['cross_attention']
    if not 0 <= detection < len(example['query_ids']):
        raise ValueError('Choose a detection from the retained boxes.')
    if not -len(layers) <= layer < len(layers):
        raise ValueError('layer must select an existing decoder layer')
    if head is not None and not 0 <= head < layers[0].shape[1]:
        raise ValueError('head must be None or select an existing attention head')
    if not 0 <= overlay <= 1:
        raise ValueError('overlay must be between 0 and 1')
    options = []
    for index, query in enumerate(example['query_ids']):
        label = example['id2label'][int(example['labels'][index])]
        options.append((f'Box {index}, q{int(query)}: {label} ({example["scores"][index]:.2f})', index))
    controls = {
        'detection': widgets.Dropdown(options=options, value=detection, description='Box:'),
        'layer': widgets.IntSlider(value=layer % len(layers), min=0, max=len(layers) - 1,
                                   description='Layer:', continuous_update=False),
        'head': widgets.Dropdown(options=[('Mean of heads', None)] + [
            (str(i), i) for i in range(layers[0].shape[1])], value=head, description='Head:'),
        'overlay': widgets.FloatSlider(value=overlay, min=0, max=1, step=0.05,
                                       description='Opacity:', continuous_update=False),
    }

    def redraw(**kwargs):
        fig = plot_detection_attention(example, **kwargs)
        plt.show()
        plt.close(fig)

    output = widgets.interactive_output(redraw, controls)
    viewer = widgets.VBox([widgets.HBox([controls['detection'], controls['head']]),
                           widgets.HBox([controls['layer'], controls['overlay']]), output])
    display(viewer)
    return viewer
