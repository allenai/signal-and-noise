import pandas as pd
from IPython.display import display, HTML

def highlight(x, as_prct=True, inverse=False):
    if inverse:
        if x > 0: style = 'color: red'
        elif x < 0: style = 'color: green'
        else: style = ''
    else:
        if x < 0: style = 'color: red'
        elif x > 0: style = 'color: green'
        else: style = ''
    if as_prct:
        text = f'<span style="{style};">{x*100:+.1f}%</span>'
    else:
        text = f'<span style="{style};">{x:+.2f}</span>'
    return text


def display_task_variants(results, key, transpose=False, as_prct=True, inverse=False, ascending=False):
    results = results.copy()
    results = results.transpose()

    # Split tasks "arc_easy:para" => ("arc_easy", "para")
    results['Task'] = results.index.str.split(':').str[0]
    results['Variant'] = results.index.str.split(':').str[1]

    # Reshape to a (Task, Variant) df
    results = results.pivot(index='Task', columns='Variant', values=key)
    results.columns = results.columns.fillna('default rc')
    results = results.infer_objects(copy=False)
    # results = results.fillna('--')
    results = results.fillna(float('-inf'))
    results = results.sort_values(by='default rc', ascending=ascending)

    # Calculate difference from "default rc" setup, display as percentage
    def format_row(x, baseline):
        diff = f' ({highlight(float(x) - baseline, as_prct=as_prct, inverse=inverse)})'
        if as_prct:
            return f"{x*100:.1f}%{diff}"
        else:
            return f"{x:.2f}{diff}"
    for col in results.columns[1:]:
        results[col] = results.apply(
            lambda row: format_row(row[col], row['default rc'])
            if row[col] != float('-inf') else '--',
            axis=1
        )
    results['default rc'] = results['default rc'].apply(
        lambda x: f"{x*100:.1f}%" if as_prct else f"{x:.2f}"
    )

    display_table(results, transpose=transpose)


def display_table(df, transpose=False, monospace=False):
    # Show as HTLM
    if transpose: df = df.transpose()
    html = df.to_html(escape=False)
    styled_html = (
        '<div style="overflow-x: auto; white-space: nowrap;">' +
        html +
        '</div>'
    )
    if monospace: styled_html = styled_html.replace('<table', '<table style="font-family: monospace"')
    styled_html = styled_html.replace('<td', '<td style="white-space: nowrap;"') # no wrap on table cells!
    display(HTML(styled_html))