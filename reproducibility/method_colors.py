"""One colour, display name and position per DE method, shared by every arm's figures.

The arms label the same method differently in their results ("DELN", "LN test",
"LN_test"; "t-test", "Scanpy t-test (log1p)"; ...), so every function here accepts
any of those labels. An unknown label raises instead of falling back, so a new
method cannot quietly get an off-palette colour or a stray name.

Legends list methods in METHODS order, LN's t-test first; figures draw LN's t-test last,
so its data sits on top of the others.
"""

LN = '#2ca02c'            # tab10 green
LOG1P = '#8c564b'         # tab10 brown
WILCOXON = '#17becf'      # tab10 cyan
MAST = '#9467bd'          # tab10 purple: tab10 orange is indistinguishable from the green under protanopia
LOG1P_ACTUAL = '#c49c94'  # tab20 light brown: the t-test's LFC recomputed on the log1p scale
SEURAT_LOG1P = '#5b3831'  # tab10 brown darkened: Seurat's log1p t-test, beside scanpy's in the CITE-seq figures
WILCOXON_UNTIED = '#9edae5'  # tab20 light cyan: Wilcoxon without tie correction, scanpy's current default, an ED check

# Box-plot medians: a plain thin line, as in the default style.
MEDIANPROPS = {'color': 'black', 'linewidth': 1.0}

# (display name, colour, labels the arms use), in drawing and legend order.
METHODS = [
    (r"LN's $t$-test", LN, ['ln', 'deln', 'ln_test', 'ln test', "ln's test", "ln's t-test",
                            "ln's-t-test", 'ziln', 'lognormal']),
    (r'log1p $t$-test', LOG1P, ['log1p', 'log1p t-test', 't-test', 't_test', 'scanpy',
                                'scanpy t-test', 'scanpy t-test (log1p)']),
    (r'log1p $t$-test (actual LFC)', LOG1P_ACTUAL, ['scanpy t-test (actual lfc)',
                                                    'log1p t-test (actual lfc)']),
    (r'Seurat log1p $t$-test', SEURAT_LOG1P, ['seurat', 'seurat_lfc', 'seurat log1p t-test']),
    ('Wilcoxon', WILCOXON, ['wilcoxon', 'scanpy wilcoxon', 'scanpy wilcoxon (log1p)']),
    ('Wilcoxon (no tie correction)', WILCOXON_UNTIED, ['wilcoxon_untied', 'wilcoxon (no tie correction)']),
    ('MAST', MAST, ['mast']),
]

_BY_KEY = {}
for _rank, (_name, _color, _aliases) in enumerate(METHODS):
    for _alias in _aliases:
        _BY_KEY[_alias] = (_rank, _name, _color)


def _lookup(label):
    key = str(label).replace('$', '').strip().lower()
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f'no standard entry for method {label!r}; add it to method_colors.py') from None


def method_color(label) -> str:
    """Colour for a method label, ignoring case and LaTeX dollar signs."""
    return _lookup(label)[2]


def method_label(label) -> str:
    """The name figures show for a method label."""
    return _lookup(label)[1]


def method_rank(label) -> int:
    """Position of a method in METHODS, for sorting."""
    return _lookup(label)[0]


def method_order(labels) -> list:
    """The distinct labels in ``labels``, in METHODS order."""
    return sorted(dict.fromkeys(labels), key=method_rank)


def method_draw_order(labels) -> list:
    """The distinct labels in METHODS order, but LN's t-test last, so it is drawn on top."""
    order = method_order(labels)
    return [m for m in order if method_rank(m) != 0] + [m for m in order if method_rank(m) == 0]


def legend_sorted(handles, labels):
    """Legend entries with the methods in METHODS order first, other entries after, as given."""
    def key(i):
        try:
            return (0, method_rank(labels[i]))
        except KeyError:
            return (1, i)
    idx = sorted(range(len(labels)), key=key)
    return [handles[i] for i in idx], [labels[i] for i in idx]
