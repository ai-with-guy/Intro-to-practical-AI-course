// Deterministic checks of the frontend's generation controls, without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, 'bertviz_local/attention_flow.js'), 'utf8');
const controlsSource = source.slice(source.indexOf('    function initializeGeneration()'),
    source.indexOf('    function renderGeneration()'));

function harness({reducedMotion = false, tokens = 3} = {}) {
    const handlers = new Map(), timeouts = new Map(), intervals = new Map();
    let timerId = 0;
    let renderCount = 0;
    const root = {
        text() { return root; },
        find(selector) {
            const element = {
                attr() { return element; },
                text() { return element; },
                on(event, callback) { handlers.set(`${selector}:${event}`, callback); return element; }
            };
            return element;
        }
    };
    const config = {rootDivId: 'test', generationStep: 0, playTimer: null};
    const context = vm.createContext({
        config, params: {generation_tokens: Array(tokens).fill('token')},
        $: () => root, renderVis() { renderCount++; },
        window: {
            setTimeout(callback) { const id = ++timerId; timeouts.set(id, callback); return id; },
            clearTimeout(id) { timeouts.delete(id); },
            setInterval(callback) { const id = ++timerId; intervals.set(id, callback); return id; },
            clearInterval(id) { intervals.delete(id); },
            matchMedia() { return {matches: reducedMotion}; }, addEventListener() {}
        }
    });
    vm.runInContext(controlsSource, context);
    context.initializeGeneration();
    return {
        config, timeouts, intervals,
        get renderCount() { return renderCount; },
        click(name) { handlers.get(`#generation-${name}:click`)(); },
        seek(step) { handlers.get('#generation-step:input')({currentTarget: {value: step}}); },
    };
}

test('one Next click appends one token immediately with exactly one UI render', () => {
    const h = harness();
    h.click('next');
    assert.equal(h.config.generationStep, 1);
    assert.equal(h.renderCount, 1);
    assert.equal(h.timeouts.size, 0);
    h.click('next');
    assert.equal(h.config.generationStep, 2);
    assert.equal(h.renderCount, 2);
    h.click('back');
    assert.equal(h.config.generationStep, 1);
    h.click('back');
    assert.equal(h.config.generationStep, 0);
    assert.equal(h.renderCount, 4);
});

test('reset and slider select a prefix without scheduling intermediate updates', () => {
    const h = harness();
    h.click('next'); h.click('reset');
    assert.equal(h.config.generationStep, 0);
    assert.equal(h.timeouts.size, 0);
    h.click('next'); h.seek(2);
    assert.equal(h.config.generationStep, 2);
    assert.equal(h.timeouts.size, 0);
});

test('playback advances one whole token per cycle and stops at the final token', () => {
    const h = harness();
    h.click('play');
    for (let step = 0; step < 3; step++) {
        const before = h.renderCount;
        h.intervals.values().next().value();
        assert.equal(h.config.generationStep, step + 1);
        assert.equal(h.renderCount, before + 1);
    }
    assert.equal(h.intervals.size, 0);
    assert.equal(h.timeouts.size, 0);
    const before = h.renderCount;
    h.click('next');
    assert.equal(h.config.generationStep, 3);
    assert.equal(h.renderCount, before);
});

test('reduced motion appends immediately, including a single-token recording', () => {
    const h = harness({reducedMotion: true, tokens: 1});
    h.click('next');
    assert.equal(h.config.generationStep, 1);
    assert.equal(h.renderCount, 1);
    assert.equal(h.timeouts.size, 0);
    h.click('back');
    assert.equal(h.config.generationStep, 0);
});
