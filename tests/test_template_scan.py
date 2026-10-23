"""Chat-template analysis (aegis/templatescan.py). Templates are parsed, never rendered.

The attacker knows these rules, so each bypass class an attacker would try next has a
case: names built at render time, hidden in variables, namespaces or macro arguments,
reached through filters, or taken from model-supplied variables such as eos_token.
"""

import pytest

from aegis.templatescan import check_template

from tests.test_model_formats import _scan_json, gguf_bytes

CHATML = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
LLAMA3 = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
)
MISTRAL = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% endif %}{% endfor %}"
)
TOOLS = (
    "{%- set ns = namespace(found=false) -%}{%- for tool in tools %}{{- tool | tojson }}"
    "{%- for name, field in tool.function.parameters.properties | items %}{{ name }}: {{ field['type'] }}"
    "{{ tool.function.parameters.properties[name] }}{%- endfor %}{%- endfor %}"
    "{%- for message in messages | reverse %}{%- if message.role == 'user' %}{% set ns.found = true %}{% endif %}"
    "{{ messages[loop.index0 - 1]['content'] }}{%- endfor %}"
)


@pytest.mark.parametrize("template", [CHATML, LLAMA3, MISTRAL, TOOLS], ids=["chatml", "llama3", "mistral", "tools"])
def test_common_chat_templates_have_no_issues(template):
    assert check_template(template) == []


BYPASSES = {
    "dunder_attribute": "{{ ''.__class__.__mro__[1].__subclasses__() }}",
    "dunder_item": "{{ cycler['__init__'] }}",
    "concatenated_name": "{% set k = '_' * 2 ~ 'globals' ~ '_' * 2 %}{{ cycler.__init__ if false }}{{ x[k] }}",
    "reversed_literal": "{{ cycler['__tini__'|reverse] }}",
    "format_built_name": "{{ cycler['%c%cinit%c%c'|format(95, 95, 95, 95)] }}",
    "namespace_attribute": "{% set ns = namespace(k='') %}{% set ns.k = '_' ~ '_init_' ~ '_' %}{{ cycler[ns.k] }}",
    "macro_argument": "{% macro get(o, k) %}{{ o[k] }}{% endmacro %}{{ get(cycler, '__init__') }}",
    "loop_over_literals": "{% for k in ['__init__'] %}{{ cycler[k] }}{% endfor %}",
    "attr_filter": "{{ cycler | attr('__init__') }}",
    "map_attribute": "{{ [cycler] | map(attribute='__init__') | list }}",
    "map_attr_filter": "{{ [cycler] | map('attr', 'x') | list }}",
    "str_format_fields": "{{ '{0.__init__.__globals__}'.format(cycler) }}",
    "model_supplied_token": "{{ cycler[eos_token] }}",
    "joined_name": "{{ cycler[['_', '_init_', '_'] | join] }}",
}


@pytest.mark.claim("C04")
@pytest.mark.parametrize("name", sorted(BYPASSES))
def test_template_injection_bypasses_are_flagged(name):
    issues = check_template(BYPASSES[name])
    assert [code for code, _ in issues] and all(code == "injection" for code, _ in issues), issues


def test_templates_that_load_other_templates_cannot_be_verified():
    assert check_template("{% include 'other.j2' %}")[0][0] == "unverifiable"
    assert check_template("{% for %}")[0][0] == "unverifiable"


def test_template_analysis_never_renders(tmp_path):
    marker = tmp_path / "rendered"
    # Would create the marker if rendered in an unsandboxed environment; parsing alone must not.
    template = ("{{ cycler.__init__.__globals__.os.popen('touch " + str(marker) + "').read() }}")
    assert check_template(template)
    assert not marker.exists()


def test_obfuscated_gguf_template_is_critical(tmp_path):
    path = tmp_path / "evil.gguf"
    path.write_bytes(gguf_bytes([("tokenizer.chat_template", BYPASSES["concatenated_name"].replace(
        "{{ cycler.__init__ if false }}", ""))], []))
    code, entry = _scan_json(path)
    assert code == 1
    assert entry["findings"][0]["rule"] == "AEGIS-G001" and entry["findings"][0]["severity"] == "critical"


def test_unparseable_gguf_template_is_a_medium_finding(tmp_path):
    path = tmp_path / "odd.gguf"
    path.write_bytes(gguf_bytes([("tokenizer.chat_template", "{% for %}")], []))
    code, entry = _scan_json(path)
    assert code == 1 and entry["findings"][0]["rule"] == "AEGIS-G003"
