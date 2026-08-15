"""
Tests for Code Analyzer.
"""
from ctxforge.engine.services.code_analyzer import CodeAnalyzer


class TestPythonAnalysis:
    """Tests for Python code analysis via AST."""

    def test_class_with_methods(self):
        code = '''
class MyService:
    """A service class."""

    def process(self, data: str) -> bool:
        """Process data."""
        return True

    async def fetch(self, url: str) -> dict:
        """Fetch data."""
        return {}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "service.py")
        assert result["language"] == "python"
        assert len(result["classes"]) == 1
        cls = result["classes"][0]
        assert cls["name"] == "MyService"
        assert cls["docstring"] == "A service class."
        assert len(cls["methods"]) == 2
        assert cls["methods"][0]["name"] == "process"
        assert cls["methods"][1]["name"] == "fetch"
        assert cls["methods"][1]["is_async"] is True

    def test_standalone_function(self):
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "greet.py")
        assert len(result["functions"]) == 1
        func = result["functions"][0]
        assert func["name"] == "hello"
        assert "name: str" in func["parameters"]
        assert func["return_type"] == "str"
        assert func["docstring"] == "Say hello."

    def test_async_function(self):
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch remote data."""
    pass
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "fetcher.py")
        assert len(result["functions"]) == 1
        assert result["functions"][0]["is_async"] is True

    def test_decorators(self):
        code = '''
import functools

@functools.lru_cache
def cached_value(key: str) -> int:
    return 42
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "cache.py")
        assert len(result["functions"]) == 1
        assert "functools.lru_cache" in result["functions"][0]["decorators"]

    def test_class_with_bases(self):
        code = '''
class Child(Parent, Mixin):
    pass
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "child.py")
        cls = result["classes"][0]
        assert cls["name"] == "Child"
        assert "Parent" in cls["base_classes"]
        assert "Mixin" in cls["base_classes"]

    def test_syntax_error_returns_empty(self):
        code = "def broken(\n"
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "broken.py")
        assert result["classes"] == []
        assert result["functions"] == []
        assert result["language"] == "python"


class TestJavaScriptAnalysis:
    """Tests for JavaScript regex analysis."""

    def test_class_declaration(self):
        code = '''
class Router extends EventEmitter {
    constructor() {}
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "router.js")
        assert result["language"] == "javascript"
        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "Router"
        assert "EventEmitter" in result["classes"][0]["base_classes"]

    def test_function_declaration(self):
        code = '''
function processData(input, options) {
    return input;
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "process.js")
        assert len(result["functions"]) >= 1
        func = result["functions"][0]
        assert func["name"] == "processData"

    def test_arrow_function(self):
        code = '''
const fetchUser = (userId) => {
    return fetch(userId);
};
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "fetch.js")
        assert len(result["functions"]) >= 1
        assert result["functions"][0]["name"] == "fetchUser"

    def test_async_function(self):
        code = '''
async function loadData(path) {
    return await read(path);
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "load.mjs")
        assert result["language"] == "javascript"
        funcs = [f for f in result["functions"] if f["name"] == "loadData"]
        assert len(funcs) == 1


class TestTypeScriptAnalysis:
    """Tests for TypeScript regex analysis."""

    def test_interface(self):
        code = '''
export interface UserConfig extends BaseConfig {
    name: string;
    age: number;
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "types.ts")
        assert result["language"] == "typescript"
        assert len(result["classes"]) >= 1
        assert result["classes"][0]["name"] == "UserConfig"

    def test_class_with_generics(self):
        code = '''
export class Repository<T> extends BaseRepo {
    async find(id: string): Promise<T> {}
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "repo.ts")
        assert len(result["classes"]) >= 1
        assert result["classes"][0]["name"] == "Repository"

    def test_typed_function(self):
        code = '''
export function parse(input: string): ParsedResult {
    return {};
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "parser.tsx")
        assert result["language"] == "typescript"
        funcs = [f for f in result["functions"] if f["name"] == "parse"]
        assert len(funcs) >= 1


class TestJavaAnalysis:
    """Tests for Java regex analysis."""

    def test_public_class(self):
        code = '''
public class UserService extends BaseService implements Serializable {
    public void save(User user) {
        // save
    }
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "UserService.java")
        assert result["language"] == "java"
        assert len(result["classes"]) >= 1
        assert result["classes"][0]["name"] == "UserService"

    def test_static_method(self):
        code = '''
public class Utils {
    public static String format(String template, Object... args) {
        return String.format(template, args);
    }
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "Utils.java")
        funcs = [f for f in result["functions"] if f["name"] == "format"]
        assert len(funcs) >= 1


class TestGoAnalysis:
    """Tests for Go regex analysis."""

    def test_struct(self):
        code = '''
type Config struct {
    Host string
    Port int
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "config.go")
        assert result["language"] == "go"
        assert len(result["classes"]) >= 1
        assert result["classes"][0]["name"] == "Config"

    def test_function(self):
        code = '''
func NewConfig(host string, port int) *Config {
    return &Config{Host: host, Port: port}
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "config.go")
        funcs = [f for f in result["functions"] if f["name"] == "NewConfig"]
        assert len(funcs) >= 1

    def test_method_on_receiver(self):
        code = '''
func (c *Config) Validate() error {
    return nil
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "config.go")
        funcs = [f for f in result["functions"] if f["name"] == "Validate"]
        assert len(funcs) >= 1


class TestRustAnalysis:
    """Tests for Rust regex analysis."""

    def test_struct(self):
        code = '''
pub struct AppConfig {
    pub host: String,
    pub port: u16,
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "config.rs")
        assert result["language"] == "rust"
        assert len(result["classes"]) >= 1
        assert result["classes"][0]["name"] == "AppConfig"

    def test_impl_block(self):
        code = '''
impl AppConfig {
    pub fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "config.rs")
        classes = [c for c in result["classes"] if c["name"] == "AppConfig"]
        assert len(classes) >= 1

    def test_async_fn(self):
        code = '''
pub async fn fetch(url: &str) -> Result<String, Error> {
    Ok(String::new())
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "http.rs")
        funcs = [f for f in result["functions"] if f["name"] == "fetch"]
        assert len(funcs) >= 1


class TestCAnalysis:
    """Tests for C/C++ regex analysis."""

    def test_c_struct(self):
        code = '''
struct Point {
    int x;
    int y;
};
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "point.c")
        assert result["language"] == "c"
        assert len(result["classes"]) >= 1
        assert result["classes"][0]["name"] == "Point"

    def test_c_function(self):
        code = '''
int add(int a, int b) {
    return a + b;
}
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "math.c")
        funcs = [f for f in result["functions"] if f["name"] == "add"]
        assert len(funcs) >= 1

    def test_cpp_class(self):
        code = '''
class Widget : public QObject {
public:
    void paint();
};
'''
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "widget.cpp")
        assert result["language"] == "cpp"
        assert len(result["classes"]) >= 1
        assert result["classes"][0]["name"] == "Widget"


class TestUnsupportedAndEdgeCases:
    """Tests for unsupported extensions and edge cases."""

    def test_unsupported_extension(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze("some content", "data.csv")
        assert result["language"] == "unknown"
        assert result["classes"] == []
        assert result["functions"] == []

    def test_supported_extension_check(self):
        assert CodeAnalyzer.supported_extension(".py") is True
        assert CodeAnalyzer.supported_extension(".js") is True
        assert CodeAnalyzer.supported_extension(".csv") is False
        assert CodeAnalyzer.supported_extension(".RS") is True  # case insensitive

    def test_is_supported_file_path(self):
        assert CodeAnalyzer.is_supported("src/main.py") is True
        assert CodeAnalyzer.is_supported("src/app.tsx") is True
        assert CodeAnalyzer.is_supported("data.json") is False

    def test_get_language(self):
        assert CodeAnalyzer.get_language("main.py") == "python"
        assert CodeAnalyzer.get_language("app.tsx") == "typescript"
        assert CodeAnalyzer.get_language("file.txt") is None

    def test_empty_content(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze("", "empty.py")
        assert result["classes"] == []
        assert result["functions"] == []


class TestFormatSummary:
    """Tests for format_summary output."""

    def test_format_summary_readable(self):
        analyzer = CodeAnalyzer()
        results = [
            {
                "file": "src/service.py",
                "language": "python",
                "classes": [{
                    "name": "UserService",
                    "base_classes": ["BaseService"],
                    "docstring": "Manages users.",
                    "methods": [{
                        "name": "create",
                        "parameters": ["self", "name: str"],
                        "return_type": "User",
                    }],
                }],
                "functions": [{
                    "name": "helper",
                    "parameters": ["x: int"],
                    "return_type": "int",
                    "docstring": "A helper.",
                }],
            },
        ]
        summary = analyzer.format_summary(results)
        assert "src/service.py" in summary
        assert "UserService" in summary
        assert "create" in summary
        assert "helper" in summary

    def test_format_summary_empty(self):
        analyzer = CodeAnalyzer()
        assert analyzer.format_summary([]) == "No code analysis available."
