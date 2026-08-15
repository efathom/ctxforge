"""
Multi-Language Code Analyzer.

Extracts class and function signatures from source code across multiple
languages. Uses Python AST for .py files and regex patterns for other
languages (JavaScript, TypeScript, Java, Go, Rust, C/C++).
"""
import ast
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_KEYWORDS = frozenset({
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
    'catch', 'try', 'finally', 'throw', 'return', 'break', 'continue',
    'sizeof', 'typeof', 'instanceof', 'new', 'delete', 'import', 'export',
    'package', 'include', 'define', 'typedef', 'using', 'namespace',
})

_LANG_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hh": "cpp",
}

_CLASS_PATTERNS: Dict[str, List[re.Pattern]] = {
    'javascript': [
        re.compile(r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+([\w.]+))?'),
    ],
    'typescript': [
        re.compile(
            r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)'
            r'(?:\s+extends\s+([\w.]+))?(?:\s+implements\s+([\w.,\s]+))?'
        ),
        re.compile(
            r'(?:export\s+)?interface\s+(\w+)'
            r'(?:\s+extends\s+([\w.,\s]+))?'
        ),
    ],
    'java': [
        re.compile(
            r'(?:public|protected|private)?\s*(?:abstract\s+|final\s+|static\s+)*'
            r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
            r'(?:\s+implements\s+([\w,\s]+))?'
        ),
        re.compile(
            r'(?:public|protected|private)?\s*interface\s+(\w+)'
            r'(?:\s+extends\s+([\w,\s]+))?'
        ),
        re.compile(r'(?:public|protected|private)?\s*enum\s+(\w+)'),
    ],
    'go': [
        re.compile(r'type\s+(\w+)\s+struct\b'),
        re.compile(r'type\s+(\w+)\s+interface\b'),
    ],
    'c': [
        re.compile(r'(?:typedef\s+)?struct\s+(\w+)'),
    ],
    'cpp': [
        re.compile(
            r'(?:template\s*<[^>]*>\s*)?class\s+(\w+)'
            r'(?:\s*:\s*(?:public|protected|private)\s+([\w:]+))?'
        ),
        re.compile(r'(?:typedef\s+)?struct\s+(\w+)'),
    ],
    'rust': [
        re.compile(r'(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)'),
        re.compile(r'(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)'),
        re.compile(r'(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)'),
        re.compile(r'impl(?:<[^>]*>)?\s+(\w+)'),
    ],
}

_FUNCTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    'javascript': [
        re.compile(
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)'
        ),
        re.compile(
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*'
            r'(?:async\s+)?\(([^)]*)\)\s*=>'
        ),
    ],
    'typescript': [
        re.compile(
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)'
            r'\s*(?:<[^>]*>)?\s*\(([^)]*)\)'
            r'(?:\s*:\s*[\w<>\[\]|&\s]+)?'
        ),
        re.compile(
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?'
            r'(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?::\s*[\w<>\[\]|&\s]+)?\s*=>'
        ),
    ],
    'java': [
        re.compile(
            r'(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?'
            r'(?:synchronized\s+)?(?:[\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)'
        ),
    ],
    'go': [
        re.compile(r'func\s+(\w+)\s*\(([^)]*)\)'),
        re.compile(r'func\s+\([^)]+\)\s+(\w+)\s*\(([^)]*)\)'),
    ],
    'c': [
        re.compile(
            r'^[\w][\w\s*]+\b(\w+)\s*\(([^)]*)\)\s*\{', re.MULTILINE
        ),
    ],
    'cpp': [
        re.compile(
            r'^[\w][\w\s*:&<>,~]+\b(\w+)\s*\(([^)]*)\)\s*'
            r'(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?\{',
            re.MULTILINE
        ),
    ],
    'rust': [
        re.compile(
            r'(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*'
            r'(?:<[^>]*>)?\s*\(([^)]*)\)'
        ),
    ],
}


class CodeAnalyzer:
    """Extract class and function signatures from source code.

    Uses Python AST for .py files and regex patterns for other languages.
    """

    @classmethod
    def supported_extension(cls, ext: str) -> bool:
        """Check if a file extension is supported."""
        return ext.lower() in _LANG_MAP

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if a file path has a supported extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in _LANG_MAP

    @classmethod
    def get_language(cls, file_path: str) -> Optional[str]:
        """Get language identifier from file path."""
        ext = os.path.splitext(file_path)[1].lower()
        return _LANG_MAP.get(ext)

    def analyze(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze source code and extract structural information.

        Routes to language-specific analyzer based on file extension.

        Args:
            content: Source code content.
            file_path: Path to the source file (used for extension detection).

        Returns:
            Dict with keys: classes, functions, language.
        """
        language = self.get_language(file_path)
        if language is None:
            return {"classes": [], "functions": [], "language": "unknown"}

        if language == "python":
            result = self._analyze_python(content)
        else:
            result = self._analyze_with_regex(content, language)

        result["language"] = language
        return result

    def _analyze_python(self, content: str) -> Dict[str, Any]:
        """Analyze Python source code using the ast module."""
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            logger.debug("Python syntax error: %s", exc)
            return {"classes": [], "functions": []}

        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self._extract_class(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._extract_function(node))

        return {"classes": classes, "functions": functions}

    @staticmethod
    def _extract_class(node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class signature from AST node."""
        base_classes: List[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name):
                    base_classes.append(f"{base.value.id}.{base.attr}")
                else:
                    base_classes.append(base.attr)

        methods: List[Dict[str, Any]] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(CodeAnalyzer._extract_function(item))

        docstring = ast.get_docstring(node)
        return {
            "name": node.name,
            "base_classes": base_classes,
            "docstring": docstring[:200] if docstring else None,
            "methods": methods,
        }

    @staticmethod
    def _extract_function(node: Any) -> Dict[str, Any]:
        """Extract function signature from AST node."""
        params: List[str] = []
        for arg in node.args.args:
            param_name = arg.arg
            if arg.annotation:
                try:
                    param_name += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(param_name)

        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                pass

        decorators: List[str] = []
        for decorator in node.decorator_list:
            try:
                decorators.append(ast.unparse(decorator))
            except Exception:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)

        docstring = ast.get_docstring(node)
        return {
            "name": node.name,
            "parameters": params,
            "return_type": return_type,
            "docstring": docstring[:200] if docstring else None,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "decorators": decorators,
        }

    def _analyze_with_regex(
        self, content: str, language: str,
    ) -> Dict[str, Any]:
        """Analyze source code using regex patterns."""
        classes = self._extract_classes_regex(content, language)
        functions = self._extract_functions_regex(content, language)
        return {"classes": classes, "functions": functions}

    @staticmethod
    def _extract_classes_regex(
        content: str, language: str,
    ) -> List[Dict[str, Any]]:
        """Extract class/struct/interface definitions via regex."""
        patterns = _CLASS_PATTERNS.get(language, [])
        classes: List[Dict[str, Any]] = []
        seen: set = set()

        for pattern in patterns:
            for match in pattern.finditer(content):
                name = match.group(1)
                if name in seen or name in _KEYWORDS:
                    continue
                seen.add(name)

                base_classes: List[str] = []
                if match.lastindex and match.lastindex >= 2 and match.group(2):
                    bases = match.group(2).strip()
                    base_classes = [
                        b.strip() for b in bases.split(',') if b.strip()
                    ]

                docstring = _extract_nearby_comment(content, match.start())
                classes.append({
                    "name": name,
                    "base_classes": base_classes,
                    "docstring": docstring[:200] if docstring else None,
                    "methods": [],
                })

        return classes

    @staticmethod
    def _extract_functions_regex(
        content: str, language: str,
    ) -> List[Dict[str, Any]]:
        """Extract function/method definitions via regex."""
        patterns = _FUNCTION_PATTERNS.get(language, [])
        functions: List[Dict[str, Any]] = []
        seen: set = set()

        for pattern in patterns:
            for match in pattern.finditer(content):
                name = match.group(1)
                if name in seen or name in _KEYWORDS:
                    continue
                seen.add(name)

                params: List[str] = []
                if match.lastindex and match.lastindex >= 2 and match.group(2):
                    raw = match.group(2).strip()
                    if raw:
                        params = [
                            p.strip() for p in raw.split(',') if p.strip()
                        ][:10]

                return_type = None
                if (match.lastindex and match.lastindex >= 3
                        and match.group(3)):
                    return_type = match.group(3).strip()

                docstring = _extract_nearby_comment(content, match.start())

                is_async = False
                line_start = content.rfind('\n', 0, match.start()) + 1
                line_prefix = content[line_start:match.start()]
                if 'async' in line_prefix:
                    is_async = True

                functions.append({
                    "name": name,
                    "parameters": params,
                    "return_type": return_type,
                    "docstring": docstring[:200] if docstring else None,
                    "is_async": is_async,
                    "decorators": [],
                })

        return functions

    def format_summary(self, results: List[Dict[str, Any]]) -> str:
        """Build a human-readable summary for LLM context.

        Args:
            results: List of analysis results from analyze().

        Returns:
            Formatted summary string.
        """
        if not results:
            return "No code analysis available."

        lines: List[str] = []
        for r in results:
            file_path = r.get("file", "unknown")
            language = r.get("language", "unknown")
            lines.append(f"## {file_path} ({language})")

            for cls in r.get("classes", []):
                bases = ", ".join(cls.get("base_classes", []))
                sig = cls["name"]
                if bases:
                    sig += f"({bases})"
                doc = cls.get("docstring") or ""
                entry = f"  Class: {sig}"
                if doc:
                    entry += f" - {doc}"
                lines.append(entry)

                for method in cls.get("methods", []):
                    mparams = ", ".join(method.get("parameters", []))
                    msig = f"{method['name']}({mparams})"
                    if method.get("return_type"):
                        msig += f" -> {method['return_type']}"
                    lines.append(f"    - {msig}")

            for func in r.get("functions", []):
                fparams = ", ".join(func.get("parameters", []))
                fsig = f"{func['name']}({fparams})"
                if func.get("return_type"):
                    fsig += f" -> {func['return_type']}"
                doc = func.get("docstring") or ""
                entry = f"  Function: {fsig}"
                if doc:
                    entry += f" - {doc}"
                lines.append(entry)

            lines.append("")

        return "\n".join(lines)


def _extract_nearby_comment(content: str, pos: int) -> Optional[str]:
    """Extract documentation comment immediately preceding the given position."""
    preceding = content[max(0, pos - 500):pos].rstrip()

    # JSDoc / JavaDoc / Doxygen block comment: /** ... */
    block_match = re.search(r'/\*\*\s*(.*?)\*/', preceding, re.DOTALL)
    if block_match:
        comment = block_match.group(1).strip()
        lines = [
            line.strip().lstrip('* ').strip()
            for line in comment.split('\n')
        ]
        return ' '.join(line for line in lines if line)[:200]

    # Line comments (// or ///)
    lines = preceding.split('\n')
    comment_lines: List[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith('//'):
            comment_lines.insert(0, stripped.lstrip('/').strip())
        elif stripped == '':
            continue
        else:
            break

    if comment_lines:
        return ' '.join(comment_lines)[:200]

    return None
