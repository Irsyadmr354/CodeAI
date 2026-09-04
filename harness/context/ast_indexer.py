import ast
import copy
import re
from typing import List

def _unparse_signature(node: ast.AST) -> List[str]:
    """Unparse a FunctionDef/ClassDef with body replaced by pass; drop trailing pass safely."""
    node_copy = copy.copy(node)
    node_copy.body = [ast.Pass()]
    ast.fix_missing_locations(node_copy)
    sig_lines = ast.unparse(node_copy).splitlines()
    # Drop the last line only if it is the synthetic 'pass' (don't assume position).
    if sig_lines and sig_lines[-1].strip() == "pass":
        sig_lines = sig_lines[:-1]
    return sig_lines

def _nested_stmt_lists(stmt: ast.AST) -> List[List[ast.stmt]]:
    """Collect nested statement lists inside a statement (if/try/with/for/...)."""
    lists: List[List[ast.stmt]] = []
    for _field, value in ast.iter_fields(stmt):
        if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
            lists.append(value)
        elif isinstance(value, ast.stmt):
            lists.append([value])
        elif isinstance(value, ast.excepthandler):
            for _f, v in ast.iter_fields(value):
                if isinstance(v, list) and v and all(isinstance(x, ast.stmt) for x in v):
                    lists.append(v)
                elif isinstance(v, ast.stmt):
                    lists.append([v])
    return lists

def _collect_nested(stmts: List[ast.stmt], indent: int = 0) -> List[str]:
    """Recursively collect func/class signatures, descending into if/try/with/loops."""
    out: List[str] = []
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.extend(_extract_node_signature(stmt, indent))
        else:
            for sub in _nested_stmt_lists(stmt):
                out.extend(_collect_nested(sub, indent))
    return out

def _extract_node_signature(node: ast.AST, indent: int = 0) -> List[str]:
    signatures = []
    prefix = " " * indent
    
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        docstring = ast.get_docstring(node)

        try:
            sig_lines = _unparse_signature(node)
        except (SyntaxError, ValueError, RecursionError):
            return [f"{prefix}def {node.name}(...)"]
        sig_text = "\n".join(sig_lines)
        
        # Prepend prefix to each line of the signature
        sig_text = "\n".join(prefix + line for line in sig_text.splitlines())
        
        if docstring:
            sig_text += f'\n{prefix}    """{docstring}"""'
            
        signatures.append(sig_text)

        # Extract nested funcs/classes at any depth (if/try/with/loop wrappers included).
        signatures.extend(_collect_nested(node.body, indent + 4))
        
    elif isinstance(node, ast.ClassDef):
        docstring = ast.get_docstring(node)

        try:
            sig_lines = _unparse_signature(node)
        except (SyntaxError, ValueError, RecursionError):
            return [f"{prefix}class {node.name}..."]
        class_sig = "\n".join(sig_lines)
        
        class_sig = "\n".join(prefix + line for line in class_sig.splitlines())
        
        if docstring:
            class_sig += f'\n{prefix}    """{docstring}"""'
            
        signatures.append(class_sig)
        
        # Extract methods and nested classes (recursive, any depth)
        signatures.extend(_collect_nested(node.body, indent + 4))
                
    return signatures

def extract_python_signatures(code: str) -> str:
    """Extract class and function signatures with docstrings from Python code."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return "Error parsing Python code."

    try:
        signatures = _collect_nested(tree.body, 0)
    except RecursionError:
        return "Error parsing Python code."
            
    return "\n\n".join(signatures)

def extract_js_ts_signatures(code: str) -> str:
    """Extract class, function, and export signatures from JS/TS code using regex."""
    signatures = []
    _js_keywords = {"if", "for", "while", "switch", "catch", "with", "else", "return", "do"}

    for raw_line in code.splitlines():
        line_clean = raw_line.strip()
        if not line_clean or line_clean.startswith("//") or line_clean.startswith("/*"):
            continue
        if line_clean.startswith("*") and not re.match(r'^\*\w+\s*\(', line_clean):
            continue
        indented = raw_line[:1].isspace()
        # Interface (incl. export / export default, generics, extends)
        if re.match(r'^(?:export\s+(?:default\s+)?)?(?:declare\s+)?interface\s+\w+', line_clean):
            signatures.append(line_clean.split('{')[0].strip())
            continue
        # Type alias (incl. export, generics)
        if re.match(r'^(?:export\s+(?:default\s+)?)?(?:declare\s+)?type\s+\w+', line_clean):
            sig = line_clean.split('{')[0].strip()
            sig = sig.rstrip(';').strip()
            signatures.append(sig)
            continue
        # Class (indent-tolerant, export/default/declare/abstract, generics/extends/implements)
        if re.match(r'^(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?class\s+\w+', line_clean):
            signatures.append(line_clean.split('{')[0].strip())
            continue
        # Function declaration (export/default/async/generator/typed-generics)
        if re.match(r'^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*\w*\s*[\(<]', line_clean):
            signatures.append(line_clean.split('{')[0].strip())
            continue
        # Arrow / function-expression assigned to var (export/default, typed, async, generics)
        if re.match(r'^(?:export\s+(?:default\s+)?)?(?:const|let|var)\s+\w[\w$]*[\w\s\?:<>\[\]\|\&\,]*=\s*(?:async\s*)?(?:\(|[\w$]+\s*=>|<)', line_clean):
            base = line_clean.split('{')[0].strip()
            if base.endswith('=>'):
                base = base.removesuffix('=>').strip() + ' => ...'
            elif '=>' in base:
                base = base.split('=>', 1)[0].strip() + ' => ...'
            else:
                base = base.rstrip(';').strip()
            signatures.append(base)
            continue
        # Class/object methods (require indent to avoid matching top-level calls):
        # [modifiers] [get|set] [*] name [<T>] (params) [: Ret] [{]
        if indented:
            m = re.match(
                r'^(?:(?:public|private|protected|static|readonly|abstract|override|async)\s+)*'
                r'(?:(?:get|set)\s+)?\*?\w+\s*(?:<[^=;{]*>)?\s*\(.*\)\s*(?::\s*[^{;=]+)?\s*\{?\s*;?\s*$',
                line_clean,
            )
            if m and '(' in line_clean and ')' in line_clean:
                first = re.match(r'^(?:(?:public|private|protected|static|readonly|abstract|override|async)\s+)*'
                                 r'(?:(?:get|set)\s+)?\*?(\w+)', line_clean)
                if first and first.group(1) not in _js_keywords:
                    signatures.append(line_clean.split('{')[0].strip().rstrip(';').strip())
                    continue
            
    return "\n".join(signatures)

class CodeSkeletonizer:
    """AST code skeletonizer for token efficiency."""
    
    @staticmethod
    def skeletonize(code: str, language: str) -> str:
        """Return the skeletonized version of the code."""
        lang = language.lower()
        if lang in ('python', 'py'):
            return extract_python_signatures(code)
        elif lang in ('javascript', 'js', 'typescript', 'ts'):
            return extract_js_ts_signatures(code)
        return "Unsupported language for skeletonization."
