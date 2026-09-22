#ifdef NDEBUG
#undef NDEBUG
#endif

#include "decision-engine.h"

#include <cassert>
#include <cmath>

static bool near(double actual, double expected) {
    return std::fabs(actual - expected) < 1e-6;
}

int main() {
    llama_decision::prefix_state_cache cache(2);
    cache.put({ 1 }, { 11 });
    cache.put({ 2 }, { 22 });
    assert(cache.size() == 2);
    assert(cache.find({ 1 }) && cache.find({ 1 })->at(0) == 11);
    cache.put({ 3 }, { 33 });
    assert(cache.size() == 2);
    assert(cache.find({ 2 }) == nullptr); // entry 1 was refreshed, so entry 2 was evicted
    assert(cache.find({ 1 }) && cache.find({ 3 }));

    llama_decision::prefix_state_cache disabled(0);
    disabled.put({ 1 }, { 11 });
    assert(disabled.size() == 0);

    const common_json schema = common_json::parse(R"({
        "route": {
            "type": "enum",
            "choices": ["dns", "forgejo", "taskboard"],
            "description": "Select a route"
        }
    })");
    const auto compiled = llama_decision::compile_schema(schema, "");

    llama_decision::result exact;
    exact.fields.push_back({ 1, 0.7f, 2, true, { 0.2f, 0.7f, 0.1f } });
    const common_json assembled = llama_decision::assemble(compiled, exact);
    const auto & field = assembled.at("fields").at("route");
    assert(field.at("value") == "forgejo");
    assert(near(field.at("probability").get<double>(), 0.7));
    assert(near(field.at("probabilities").at("dns").get<double>(), 0.2));
    assert(near(field.at("probabilities").at("forgejo").get<double>(), 0.7));
    assert(near(field.at("probabilities").at("taskboard").get<double>(), 0.1));

    llama_decision::result greedy;
    greedy.fields.push_back({ 1, 0.7f, 1, false, {} });
    const common_json partial = llama_decision::assemble(compiled, greedy);
    assert(!partial.at("fields").at("route").contains("probabilities"));
}
