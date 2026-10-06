/**
 * @fileoverview Attention flow view, derived from the BertViz head view.
 *
 *
 *  Based on: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/visualization/attention.js
 *
 * Change log:
 *
 * 12/19/18  Jesse Vig   Assorted cleanup. Changed orientation of attention matrices.
 * 12/29/20  Jesse Vig   Significant refactor.
 * 12/31/20  Jesse Vig   Support multiple visualizations in single notebook.
 * 02/06/21  Jesse Vig   Move require config from separate jupyter notebook step
 * 05/03/21  Jesse Vig   Adjust height of visualization dynamically
 * 07/25/21  Jesse Vig   Support layer filtering
 * 03/23/22  Daniel SC   Update requirement URLs for d3 and jQuery (source of bug not allowing end result to be displayed on browsers)
 **/

(function ($, d3) {

    const params = window.BERTVIZ_PARAMS;
    const TEXT_SIZE = 15;
    const BOXWIDTH = 110;
    const BOXHEIGHT = 22.5;
    const MATRIX_WIDTH = 115;
    const CHECKBOX_SIZE = 20;
    const TEXT_TOP = 52;
    const MAX_FLOW_DOTS = 300;

    console.log("d3 version", d3.version)
    let headColors;
    try {
        headColors = d3.scaleOrdinal(d3.schemeCategory10);
    } catch (err) {
        console.log('Older d3 version')
        headColors = d3.scale.category10();
    }
    let config = {};
    initialize();
    renderVis();

    function initialize() {
        config.attention = params['attention'];
        config.filter = params['default_filter'];
        config.rootDivId = params['root_div_id'];
        config.nLayers = config.attention[config.filter]['attn'].length;
        config.nHeads = config.attention[config.filter]['attn'][0].length;
        config.layers = params['include_layers']
        config.generationStep = 0;
        config.generationPhase = 0; // Read prefix, predict next token, append token.
        config.playTimer = null;
        config.animate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        $(`#${config.rootDivId} #animate-flow`).prop('checked', config.animate)
            .on('change', function (e) {
                config.animate = e.currentTarget.checked;
                updateAnimation(d3.select(`#${config.rootDivId} #vis svg`));
            });

        if (params['heads']) {
            config.headVis = new Array(config.nHeads).fill(false);
            params['heads'].forEach(x => config.headVis[x] = true);
        } else {
            config.headVis = new Array(config.nHeads).fill(true);
        }
        config.initialTextLength = config.attention[config.filter].right_text.length;
        config.layer_seq = (params['layer'] == null ? 0 : config.layers.findIndex(layer => params['layer'] === layer));
        config.layer = config.layers[config.layer_seq]

        let layerEl = $(`#${config.rootDivId} #layer`);
        for (const layer of config.layers) {
            layerEl.append($("<option />").val(layer).text(layer));
        }
        layerEl.val(config.layer).change();
        layerEl.on('change', function (e) {
            config.layer = +e.currentTarget.value;
            config.layer_seq = config.layers.findIndex(layer => config.layer === layer);
            renderVis();
        });

        $(`#${config.rootDivId} #filter`).val(config.filter).on('change', function (e) {
            config.filter = e.currentTarget.value;
            const nHeads = config.attention[config.filter].attn[0].length;
            if (nHeads !== config.nHeads) {
                config.nHeads = nHeads;
                config.headVis = new Array(nHeads).fill(true);
            }
            renderVis();
        });
        if (params.generation_tokens) initializeGeneration();
    }

    function initializeGeneration() {
        const root = $(`#${config.rootDivId}`);
        root.find('#generation-step').attr('max', params.generation_tokens.length - 1)
            .on('input', function (e) {
                stopPlayback();
                config.generationStep = +e.currentTarget.value;
                config.generationPhase = 0;
                renderVis();
            });
        root.find('#generation-next').on('click', function () {
            stopPlayback();
            advanceGeneration();
        });
        root.find('#generation-back').on('click', function () {
            stopPlayback();
            if (config.generationPhase > 0) config.generationPhase--;
            else if (config.generationStep > 0) {
                config.generationStep--;
                config.generationPhase = 2;
            }
            renderVis();
        });
        root.find('#generation-reset').on('click', function () {
            stopPlayback();
            config.generationStep = 0;
            config.generationPhase = 0;
            renderVis();
        });
        root.find('#generation-play').on('click', function () {
            if (config.playTimer !== null) stopPlayback();
            else {
                if (generationFinished()) {
                    config.generationStep = 0;
                    config.generationPhase = 0;
                }
                config.playTimer = window.setInterval(advanceGeneration, 1000);
                root.find('#generation-play').text('Pause');
                renderVis();
            }
        });
        window.addEventListener('pagehide', stopPlayback);
    }

    function generationFinished() {
        return config.generationStep === params.generation_tokens.length - 1 && config.generationPhase === 2;
    }

    function stopPlayback() {
        if (config.playTimer !== null) window.clearInterval(config.playTimer);
        config.playTimer = null;
        $(`#${config.rootDivId} #generation-play`).text('Play');
    }

    function advanceGeneration() {
        if (generationFinished()) {
            stopPlayback();
            return;
        }
        if (config.generationPhase < 2) config.generationPhase++;
        else {
            config.generationStep++;
            config.generationPhase = 0;
        }
        if (generationFinished()) stopPlayback();
        renderVis();
    }

    function renderGeneration() {
        const root = $(`#${config.rootDivId}`);
        const decoder = config.attention.find(d => d.name === 'Cross' || d.name === 'Decoder');
        const prefix = decoder.left_text.slice(0, config.generationStep + 1);
        const next = params.generation_tokens[config.generationStep];
        if (config.generationPhase === 2) prefix.push(next);
        const stream = root.find('#generation-stream').empty();
        prefix.forEach((token, i) => $('<span />').text(token).css({
            display: 'inline-block', padding: '4px 7px', margin: '2px', borderRadius: '4px',
            background: i === prefix.length - 1 ? '#c9e3ff' : '#e3e8ef', fontFamily: 'monospace'
        }).appendTo(stream));
        const phases = ['Read prefix → attend to available tokens',
                        'Predict the next token from the final query',
                        'Append the prediction to the decoder stream'];
        root.find('#generation-status').text(`Step ${config.generationStep + 1}/${params.generation_tokens.length}: ${phases[config.generationPhase]}`);
        root.find('#generation-prediction').text(config.generationPhase === 0
            ? 'Next token: not revealed yet'
            : config.generationPhase === 1 ? `Next token: ${next}`
            : generationFinished() ? `Appended ${next}. End of recorded generation (EOS or token limit).`
            : `Appended ${next}. Next: read the extended prefix to predict again.`);
        root.find('#generation-step').val(config.generationStep);
        root.find('#generation-next').prop('disabled', generationFinished());
        root.find('#generation-back').prop('disabled', config.generationStep === 0 && config.generationPhase === 0);
    }

    function renderVis() {

        // Load parameters
        const attnData = config.attention[config.filter];
        let leftText = attnData.left_text;
        let rightText = attnData.right_text;

        // Select attention for given layer
        let layerAttention = attnData.attn[config.layer_seq];
        if (params.generation_tokens) {
            renderGeneration();
            const prefixLength = config.generationStep + 1;
            if (attnData.name === 'Cross' || attnData.name === 'Decoder') {
                leftText = leftText.slice(0, prefixLength);
                layerAttention = layerAttention.map(head => head.slice(0, prefixLength));
            }
            if (attnData.name === 'Decoder') {
                rightText = rightText.slice(0, prefixLength);
                layerAttention = layerAttention.map(head => head.map(row => row.slice(0, prefixLength)));
            }
        }

        // Clear vis
        $(`#${config.rootDivId} #vis`).empty();

        // Determine size of visualization
        const height = Math.max(leftText.length, rightText.length) * BOXHEIGHT + TEXT_TOP;
        const svg = d3.select(`#${config.rootDivId} #vis`)
            .append('svg')
            .attr("width", "100%")
            .attr("height", height + "px");

        // Display tokens on left and right side of visualization
        renderText(svg, leftText, true, layerAttention, 0);
        renderText(svg, rightText, false, layerAttention, MATRIX_WIDTH + BOXWIDTH);

        // Render attention arcs
        renderAttention(svg, layerAttention);

        // Draw squares at top of visualization, one for each head
        drawCheckboxes(0, svg, layerAttention);
        const labels = attnData.name === 'Cross'
            ? ['Decoder queries (receive)', 'Encoder keys/values (send)']
            : ['Queries (receive)', 'Keys/values (send)'];
        labels.forEach((label, i) => svg.append('text')
            .attr('x', i ? MATRIX_WIDTH + BOXWIDTH : 0)
            .attr('y', 43).attr('font-size', '11px').attr('fill', '#555').text(label));
        focusGeneration(svg);
    }

    function focusGeneration(svg) {
        if (!params.generation_tokens || config.attention[config.filter].name === 'Encoder') return;
        // Only the final available query predicts the next token at this step.
        svg.select('#attention').attr('visibility', 'hidden');
        svg.selectAll(`.attentionEdge[left-token-index='${config.generationStep}']`)
            .attr('visibility', 'visible');
        svg.select('#left').selectAll('.background')
            .style('opacity', (d, i) => i === config.generationStep ? 0.5 : 0);
    }

    function renderText(svg, text, isLeft, attention, leftPos) {

        const textContainer = svg.append("svg:g")
            .attr("id", isLeft ? "left" : "right");

        // Add attention highlights superimposed over words
        textContainer.append("g")
            .classed("attentionBoxes", true)
            .selectAll("g")
            .data(attention)
            .enter()
            .append("g")
            .attr("head-index", (d, i) => i)
            .selectAll("rect")
            .data(d => isLeft ? d : transpose(d)) // if right text, transpose attention to get right-to-left weights
            .enter()
            .append("rect")
            .attr("x", function () {
                var headIndex = +this.parentNode.getAttribute("head-index");
                return leftPos + boxOffsets(headIndex);
            })
            .attr("y", (+1) * BOXHEIGHT)
            .attr("width", BOXWIDTH / activeHeads())
            .attr("height", BOXHEIGHT)
            .attr("fill", function () {
                return headColors(+this.parentNode.getAttribute("head-index"))
            })
            .style("opacity", 0.0);

        const tokenContainer = textContainer.append("g").selectAll("g")
            .data(text)
            .enter()
            .append("g");

        // Add gray background that appears when hovering over text
        tokenContainer.append("rect")
            .classed("background", true)
            .style("opacity", 0.0)
            .attr("fill", "lightgray")
            .attr("x", leftPos)
            .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT)
            .attr("width", BOXWIDTH)
            .attr("height", BOXHEIGHT);

        // Add token text
        const textEl = tokenContainer.append("text")
            .text(d => d)
            .attr("font-size", TEXT_SIZE + "px")
            .style("cursor", "default")
            .style("-webkit-user-select", "none")
            .attr("x", leftPos)
            .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT);

        if (isLeft) {
            textEl.style("text-anchor", "end")
                .attr("dx", BOXWIDTH - 0.5 * TEXT_SIZE)
                .attr("dy", TEXT_SIZE);
        } else {
            textEl.style("text-anchor", "start")
                .attr("dx", +0.5 * TEXT_SIZE)
                .attr("dy", TEXT_SIZE);
        }

        tokenContainer.on("mouseover", function (d, index) {

            // Show gray background for moused-over token
            textContainer.selectAll(".background")
                .style("opacity", (d, i) => i === index ? 1.0 : 0.0)

            // Reset visibility attribute for any previously highlighted attention arcs
            svg.select("#attention")
                .selectAll(".attentionEdge[visibility='visible']")
                .attr("visibility", null)

            // Hide group containing attention arcs
            svg.select("#attention").attr("visibility", "hidden");

            // Set to visible appropriate attention arcs to be highlighted
            if (isLeft) {
                svg.select("#attention").selectAll(".attentionEdge[left-token-index='" + index + "']").attr("visibility", "visible");
            } else {
                svg.select("#attention").selectAll(".attentionEdge[right-token-index='" + index + "']").attr("visibility", "visible");
            }

            // Update color boxes superimposed over tokens
            const id = isLeft ? "right" : "left";
            const leftPos = isLeft ? MATRIX_WIDTH + BOXWIDTH : 0;
            svg.select("#" + id)
                .selectAll(".attentionBoxes")
                .selectAll("g")
                .attr("head-index", (d, i) => i)
                .selectAll("rect")
                .attr("x", function () {
                    const headIndex = +this.parentNode.getAttribute("head-index");
                    return leftPos + boxOffsets(headIndex);
                })
                .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT)
                .attr("width", BOXWIDTH / activeHeads())
                .attr("height", BOXHEIGHT)
                .style("opacity", function (d) {
                    const headIndex = +this.parentNode.getAttribute("head-index");
                    if (config.headVis[headIndex])
                        if (d) {
                            return d[index];
                        } else {
                            return 0.0;
                        }
                    else
                        return 0.0;
                });
        });

        textContainer.on("mouseleave", function () {

            // Unhighlight selected token
            d3.select(this).selectAll(".background")
                .style("opacity", 0.0);

            // Reset visibility attributes for previously selected lines
            svg.select("#attention")
                .selectAll(".attentionEdge[visibility='visible']")
                .attr("visibility", null) ;
            svg.select("#attention").attr("visibility", "visible");

            // Reset highlights superimposed over tokens
            svg.selectAll(".attentionBoxes")
                .selectAll("g")
                .selectAll("rect")
                .style("opacity", 0.0);
            focusGeneration(svg);
        });
    }

    function renderAttention(svg, attention) {

        // Remove previous dom elements
        svg.select("#attention").remove();

        // Add new elements
        svg.append("g")
            .attr("id", "attention") // Container for all attention arcs
            .selectAll(".headAttention")
            .data(attention)
            .enter()
            .append("g")
            .classed("headAttention", true) // Group attention arcs by head
            .attr("head-index", (d, i) => i)
            .selectAll(".tokenAttention")
            .data(d => d)
            .enter()
            .append("g")
            .classed("tokenAttention", true) // Group attention arcs by left token
            .attr("left-token-index", (d, i) => i)
            .selectAll(".attentionEdge")
            .data(d => d)
            .enter()
            .append("g")
            .classed("attentionEdge", true)
            .attr("left-token-index", function () {
                return +this.parentNode.getAttribute("left-token-index");
            })
            .attr("right-token-index", (d, i) => i)
            .append("line")
            .attr("x1", BOXWIDTH)
            .attr("y1", function () {
                const leftTokenIndex = +this.parentNode.getAttribute("left-token-index")
                return TEXT_TOP + leftTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2)
            })
            .attr("x2", BOXWIDTH + MATRIX_WIDTH)
            .attr("y2", (d, rightTokenIndex) => TEXT_TOP + rightTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2))
            .attr("stroke-width", 2)
            .attr("stroke", function () {
                const headIndex = +this.parentNode.parentNode.parentNode.getAttribute("head-index");
                return headColors(headIndex)
            })
            .attr("left-token-index", function () {
                return +this.parentNode.getAttribute("left-token-index")
            })
            .attr("right-token-index", (d, i) => i)
        ;
        updateAttention(svg)
    }

    function updateAttention(svg) {
        svg.select("#attention")
            .selectAll("line")
            .attr("stroke-opacity", function (d) {
                const headIndex = +this.parentNode.parentNode.parentNode.getAttribute("head-index");
                // If head is selected
                if (config.headVis[headIndex]) {
                    // Set opacity to attention weight divided by number of active heads
                    return d / activeHeads()
                } else {
                    return 0.0;
                }
            });
        renderFlowDots(svg);
    }

    function renderFlowDots(svg) {
        svg.selectAll('.flowDot').remove();
        const edges = [];
        svg.selectAll('.attentionEdge').each(function (weight) {
            const head = +this.parentNode.parentNode.getAttribute('head-index');
            if (config.headVis[head] && weight > 0) edges.push({node: this, weight, head});
        });
        // Bound animated elements for long sequences; all attention lines remain.
        const isCurrentQuery = edge => params.generation_tokens
            && config.attention[config.filter].name !== 'Encoder'
            && +edge.node.getAttribute('left-token-index') === config.generationStep;
        edges.sort((a, b) => Number(isCurrentQuery(b)) - Number(isCurrentQuery(a)) || b.weight - a.weight);
        edges.slice(0, MAX_FLOW_DOTS).forEach(({node, weight, head}, i) => {
            const line = node.querySelector('line');
            const x1 = line.getAttribute('x1'), y1 = line.getAttribute('y1');
            const x2 = line.getAttribute('x2'), y2 = line.getAttribute('y2');
            const dot = d3.select(node).append('circle').classed('flowDot', true)
                .attr('r', 2.5).attr('fill', headColors(head))
                .attr('fill-opacity', Math.min(1, weight * 2 / activeHeads()))
                .attr('pointer-events', 'none');
            // Attention rows are queries; values flow from column to row, not vice versa.
            dot.append('animateMotion')
                .attr('path', `M ${x2} ${y2} L ${x1} ${y1}`)
                .attr('dur', '1.8s').attr('begin', `${-(i % 19) * 0.1}s`)
                .attr('repeatCount', 'indefinite');
        });
        updateAnimation(svg);
    }

    function updateAnimation(svg) {
        const node = svg.node();
        if (!node) return;
        svg.selectAll('.flowDot').style('display', config.animate ? null : 'none');
        if (config.animate) node.unpauseAnimations();
        else node.pauseAnimations();
    }

    function boxOffsets(i) {
        const numHeadsAbove = config.headVis.reduce(
            function (acc, val, cur) {
                return val && cur < i ? acc + 1 : acc;
            }, 0);
        return numHeadsAbove * (BOXWIDTH / activeHeads());
    }

    function activeHeads() {
        return config.headVis.reduce(function (acc, val) {
            return val ? acc + 1 : acc;
        }, 0);
    }

    function drawCheckboxes(top, svg) {
        const checkboxContainer = svg.append("g");
        const checkbox = checkboxContainer.selectAll("rect")
            .data(config.headVis)
            .enter()
            .append("rect")
            .attr("fill", (d, i) => headColors(i))
            .attr("x", (d, i) => i * CHECKBOX_SIZE)
            .attr("y", top)
            .attr("width", CHECKBOX_SIZE)
            .attr("height", CHECKBOX_SIZE);

        function updateCheckboxes() {
            checkboxContainer.selectAll("rect")
                .data(config.headVis)
                .attr("fill", (d, i) => d ? headColors(i): lighten(headColors(i)));
        }

        updateCheckboxes();

        checkbox.on("click", function (d, i) {
            if (config.headVis[i] && activeHeads() === 1) return;
            config.headVis[i] = !config.headVis[i];
            updateCheckboxes();
            updateAttention(svg);
        });

        checkbox.on("dblclick", function (d, i) {
            // If we double click on the only active head then reset
            if (config.headVis[i] && activeHeads() === 1) {
                config.headVis = new Array(config.nHeads).fill(true);
            } else {
                config.headVis = new Array(config.nHeads).fill(false);
                config.headVis[i] = true;
            }
            updateCheckboxes();
            updateAttention(svg);
        });
    }

    function lighten(color) {
        const c = d3.hsl(color);
        const increment = (1 - c.l) * 0.6;
        c.l += increment;
        c.s -= increment;
        return c;
    }

    function transpose(mat) {
        return mat[0].map(function (col, i) {
            return mat.map(function (row) {
                return row[i];
            });
        });
    }

})(window.jQuery, window.d3);
