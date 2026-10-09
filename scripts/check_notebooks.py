"""Validate complete notebooks and embedded source without executing models."""
import ast
import json
from pathlib import Path


def main():
    import nbformat
    count = 0
    for path in sorted(Path('notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            tree = ast.parse(cell.source, filename=f'{path}:cell{index}')
            compile(tree, f'{path}:cell{index}', 'exec')
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and node.func.attr == 'write_text' and node.args \
                        and isinstance(node.args[0], ast.Constant) \
                        and isinstance(node.args[0].value, str) \
                        and node.args[0].value.startswith('"""'):
                    compile(node.args[0].value, f'{path}:embedded{index}', 'exec')
        count += 1
        print(f'PASS {path}')
    assert count >= 2, 'Production and qualification notebooks must be present'
    print(json.dumps({'notebooks': count, 'status': 'PASS'}))


if __name__ == '__main__':
    main()
