import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

type Mode = 'test' | 'nextest' | 'coverage' | 'clippy' | 'live';

export interface Invocation {
  command: string;
  args: string[];
  env: NodeJS.ProcessEnv;
  cwd: string;
}

const profileVariables = new Set([
  'HOME', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'APPDATA', 'LOCALAPPDATA',
  'XDG_DATA_HOME', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'TEMP', 'TMP', 'TMPDIR',
  'CODEX_HOME', 'COPILOT_CLI_HOME', 'CLAUDE_CONFIG_DIR', 'CLAUDE_POWERPOINT_INDEXEDDB_PATH',
  'CONTINUE_GLOBAL_DIR', 'FORGE_CONFIG', 'GEMINI_HOME', 'GOOSE_PATH_ROOT',
  'HERMES_HOME', 'HERMES_DATA_DIR_SUFFIX', 'INTERPRETER_HOME', 'KIMI_SHARE_DIR',
  'KIMI_HOME', 'LLM_USER_PATH', 'OPENCODE_HOME', 'QWEN_RUNTIME_DIR', 'QWEN_HOME',
  'VIBE_HOME', 'CCHV_TEST_VSCODE_USER_DATA_ROOT', 'CCHV_TOKEN', 'FEEDBACK_EMAIL',
]);

export function buildInvocation(
  input: string[],
  inheritedEnv: NodeJS.ProcessEnv,
  fixtureRoot: string,
  repositoryRoot: string,
): Invocation {
  const options = [...input];
  const mode = (options[0] && !options[0].startsWith('-') ? options.shift() : 'test') as Mode;
  if (!['test', 'nextest', 'coverage', 'clippy', 'live'].includes(mode)) {
    throw new Error('Use test, nextest, coverage, clippy, or live --acknowledge-profile-access.');
  }
  let filter: string | undefined;
  let features: string | undefined;
  let profile: string | undefined;
  let testTarget: string | undefined;
  let library = false;
  let allFeatures = false;
  let open = false;
  let lcov = false;
  let acknowledgeLive = false;
  const seen = new Set<string>();
  while (options.length > 0) {
    const option = options.shift()!;
    if (seen.has(option)) throw new Error(`Duplicate argument: ${option}`);
    seen.add(option);
    const value = (): string => {
      const result = options.shift();
      if (!result || result.startsWith('-')) throw new Error(`Missing or unsafe value for ${option}`);
      return result;
    };
    switch (option) {
      case '--filter': filter = value(); break;
      case '--features':
        features = value();
        if (features !== 'webui-server') throw new Error('Only the webui-server feature is supported.');
        break;
      case '--profile':
        profile = value();
        if (!['ci', 'default', 'quick', 'coverage'].includes(profile)) throw new Error('Unsupported nextest profile.');
        break;
      case '--test':
        testTarget = value();
        if (!/^[a-zA-Z0-9_]+$/.test(testTarget) || testTarget === 'live_profile') {
          throw new Error('Use the explicit live mode for live-profile diagnostics.');
        }
        break;
      case '--lib': library = true; break;
      case '--all-features': allFeatures = true; break;
      case '--open': open = true; break;
      case '--lcov': lcov = true; break;
      case '--acknowledge-profile-access': acknowledgeLive = true; break;
      default: throw new Error(`Unsupported or unsafe argument: ${option}`);
    }
  }
  if (features && allFeatures) throw new Error('Choose --features or --all-features.');
  if (library && testTarget) throw new Error('Choose --lib or --test.');
  if (profile && mode !== 'nextest' && mode !== 'coverage') throw new Error('--profile requires nextest or coverage.');
  if (open && mode !== 'coverage') throw new Error('--open requires coverage.');
  if (lcov && (mode !== 'coverage' || open)) throw new Error('--lcov requires coverage without --open.');
  if ((library || testTarget) && mode !== 'test') throw new Error('--lib and --test require test mode.');
  if (mode === 'clippy' && (filter || features || allFeatures)) throw new Error('Clippy always checks all targets and features.');
  if (mode === 'live' && (!acknowledgeLive || seen.size !== 1)) {
    throw new Error('Live diagnostics require exactly live --acknowledge-profile-access; they can read profiles and write derived caches.');
  }
  if (acknowledgeLive && mode !== 'live') throw new Error('Profile access acknowledgement requires live mode.');

  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(inheritedEnv)) {
    const name = key.toUpperCase();
    if ((name === 'RUSTC_WRAPPER' || name === 'RUSTC_WORKSPACE_WRAPPER' || name === 'CARGO_BUILD_TARGET' || /^CARGO_TARGET_.*_RUNNER$/.test(name)) && value) {
      throw new Error(`Remove alternate build/run override ${key} before using the safe runner.`);
    }
    if (name.startsWith('CARGO_ALIAS_') && value) throw new Error(`Remove Cargo alias override ${key} before using the safe runner.`);
    if ((name === 'RUSTFLAGS' || name === 'CARGO_BUILD_RUSTFLAGS' || /^CARGO_TARGET_.*_RUSTFLAGS$/.test(name)) && value && !/^-D\s*warnings$/.test(value)) {
      throw new Error(`Remove compiler override ${key} before using the safe runner.`);
    }
    if (name === 'CARGO_ENCODED_RUSTFLAGS' && value) throw new Error('Remove CARGO_ENCODED_RUSTFLAGS before using the safe runner.');
    if (name === 'CCHV_TEST_PROFILE' || name === 'CARGO_TARGET_DIR' || name === 'RUST_TEST_THREADS' || name.startsWith('NEXTEST_')) continue;
    if (mode !== 'live' && profileVariables.has(name)) continue;
    env[key] = value;
  }
  env['CCHV_TEST_PROFILE'] = mode === 'live' ? 'live-v1' : 'isolated-v1';
  env['CARGO_TARGET_DIR'] = join(repositoryRoot, 'src-tauri', 'target', mode === 'live' ? 'test-live' : 'test-isolated');
  env['RUST_TEST_THREADS'] = '1';
  if (mode !== 'live') {
    Object.assign(env, {
      HOME: join(fixtureRoot, 'home'), USERPROFILE: join(fixtureRoot, 'home'),
      APPDATA: join(fixtureRoot, 'config'), LOCALAPPDATA: join(fixtureRoot, 'data'),
      XDG_CONFIG_HOME: join(fixtureRoot, 'config'), XDG_DATA_HOME: join(fixtureRoot, 'data'),
      XDG_CACHE_HOME: join(fixtureRoot, 'cache'),
      TEMP: join(fixtureRoot, 'tmp'), TMP: join(fixtureRoot, 'tmp'), TMPDIR: join(fixtureRoot, 'tmp'),
    });
  }
  let args: string[];
  if (mode === 'live') args = ['test', '--test', 'live_profile', '--', '--ignored', '--test-threads=1'];
  else if (mode === 'clippy') args = ['clippy', '--all-targets', '--all-features', '--', '-D', 'warnings'];
  else {
    args = mode === 'test' ? ['test'] : mode === 'nextest' ? ['nextest', 'run'] : ['llvm-cov', 'nextest', ...(lcov ? ['--lcov', '--output-path', 'lcov.info'] : ['--html'])];
    if (features) args.push('--features', features);
    if (allFeatures) args.push('--all-features');
    if (profile) args.push('--profile', profile);
    if (open) args.push('--open');
    if (library) args.push('--lib');
    if (testTarget) args.push('--test', testTarget);
    if (filter) args.push(filter);
    if (mode === 'test') args.push('--', '--test-threads=1');
  }
  return { command: 'cargo', args, env, cwd: join(repositoryRoot, 'src-tauri') };
}

// Preserve offsets while excluding comments and literals from architectural checks.
function rustCode(source: string): string {
  const parts: string[] = [];
  let plainStart = 0;
  let index = 0;
  const mask = (end: number): void => {
    parts.push(source.slice(plainStart, index), source.slice(index, end).replace(/[^\r\n]/g, ' '));
    index = end;
    plainStart = end;
  };
  while (index < source.length) {
    if (source.startsWith('//', index)) {
      const end = source.indexOf('\n', index + 2);
      mask(end < 0 ? source.length : end);
      continue;
    }
    if (source.startsWith('/*', index)) {
      let end = index + 2;
      let depth = 1;
      while (end < source.length && depth > 0) {
        if (source.startsWith('/*', end)) { depth++; end += 2; }
        else if (source.startsWith('*/', end)) { depth--; end += 2; }
        else end++;
      }
      if (depth !== 0) throw new Error('Unterminated Rust block comment.');
      mask(end);
      continue;
    }
    if ((source[index] === 'r' || source.startsWith('br', index)) && (index === 0 || !/[\w]/.test(source[index - 1]!))) {
      const raw = /^(?:br|r)(#*)"/.exec(source.slice(index));
      if (raw) {
        const closing = `"${raw[1]}`;
        const end = source.indexOf(closing, index + raw[0].length);
        if (end < 0) throw new Error('Unterminated Rust raw string.');
        mask(end + closing.length);
        continue;
      }
    }
    if (source[index] === "'" || source.startsWith("b'", index)) {
      const character = /^(?:b)?'(?:\\(?:u\{[\da-fA-F_]+\}|x[\da-fA-F]{2}|[^\r\n])|[^'\\\r\n])'/u.exec(source.slice(index));
      if (character) { mask(index + character[0].length); continue; }
    }
    if (source[index] === '"') {
      let end = index + 1;
      while (end < source.length && source[end] !== '"') end += source[end] === '\\' ? 2 : 1;
      if (end >= source.length) throw new Error('Unterminated Rust string.');
      mask(end + 1);
      continue;
    }
    index++;
  }
  return [...parts, source.slice(plainStart)].join('');
}

function outsideProfileAdapter(code: string): string {
  return code.replace(/\b(?:crate|claude_code_history_viewer_lib)\s*::\s*profile_paths\s*::\s*env\s*::\s*(?:var|var_os|set_var|remove_var)\b/g, 'scoped_profile_environment');
}

function checkEnvironmentSyntax(source: string, path: string): void {
  const code = rustCode(source);
  for (const statement of code.matchAll(/\buse\b[^;]*;/g)) {
    if (/\benv\b/.test(statement[0])) throw new Error(`Environment imports must use the fully qualified profile adapter: ${path}`);
  }
  for (const macro of code.matchAll(/\b(?:env|option_env)\s*!\s*\(/g)) {
    const argument = source.slice(macro.index + macro[0].length);
    if (!/^\s*"(?:CARGO_PKG_VERSION|CARGO_MANIFEST_DIR)"\s*\)/.test(argument)) {
      throw new Error(`Compile-time environment access is limited to reviewed Cargo metadata: ${path}`);
    }
  }
}

const directoryFunctions = new Set([
  'home_dir', 'data_dir', 'data_local_dir', 'config_dir',
  'cache_dir', 'download_dir', 'document_dir', 'desktop_dir',
]);

function directoryCompatibilityPrefix(code: string): string | undefined {
  const prefix = /^\s*extern\s+crate\s+self\s+as\s+dirs\s*;\s*(?:#\s*\[\s*allow\s*\(\s*unused_imports\s*\)\s*\]\s*)?pub\s*\(\s*crate\s*\)\s+use\s+crate\s*::\s*profile_paths\s*::\s*\{([^{}]*)\}\s*;/.exec(code);
  if (!prefix) return undefined;
  const names = prefix[1]!.split(',').map(name => name.trim()).filter(Boolean);
  return names.length === directoryFunctions.size && new Set(names).size === names.length
    && names.every(name => directoryFunctions.has(name)) ? prefix[0] : undefined;
}

function withoutDirectoryCompatibility(code: string): string {
  return code.replace(/\bdirs\s*::\s*(\w+|\{[^{}]*\})/g, (original: string, member: string, offset: number) => {
    if (code.slice(0, offset).trimEnd().endsWith('::') || code[offset - 1] === '#') return original;
    const names = member.startsWith('{')
      ? member.slice(1, -1).split(',').map(name => name.trim()).filter(Boolean)
      : [member];
    return names.length > 0 && names.every(name => {
      const imported = /^(\w+)(?:\s+as\s+\w+)?$/.exec(name);
      return imported !== null && directoryFunctions.has(imported[1]!);
    }) ? 'scoped_profile_directory' : original;
  });
}

function checkDirectoryBindings(code: string, path: string): void {
  code = code.replace(/\br#dirs\b/g, 'dirs');
  if (/\b(?:mod|type|struct|enum|trait|fn|const|static)\s+dirs\b|\bextern\s+crate\s+dirs\b|\b(?:extern\s+crate|use)\b[^;]*\bas\s+dirs\b/.test(code)) {
    throw new Error(`Competing directory compatibility binding: ${path}`);
  }
  for (const statement of code.matchAll(/\buse\b[^;]*;/g)) {
    if (/\bdirs\b/.test(statement[0]) && !/^use\s+dirs\s*::/.test(statement[0])) {
      throw new Error(`Competing directory compatibility import: ${path}`);
    }
  }
}

function rustFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Test source cannot be a symbolic link: ${path}`);
    return entry.isDirectory() ? rustFiles(path) : entry.name.endsWith('.rs') ? [path] : [];
  });
}

export function checkTestBoundary(repositoryRoot: string): void {
  const sourceRoot = join(repositoryRoot, 'src-tauri', 'src');
  const libraryRoot = join(sourceRoot, 'lib.rs');
  const libraryCode = existsSync(libraryRoot) ? rustCode(readFileSync(libraryRoot, 'utf8')) : '';
  const compatibilityPrefix = directoryCompatibilityPrefix(libraryCode);
  const manifestPath = join(repositoryRoot, 'src-tauri', 'Cargo.toml');
  if (existsSync(manifestPath)) {
    const manifest = readFileSync(manifestPath, 'utf8').replace(/^\s*#.*$/gm, '');
    const withoutNativeDependency = manifest.replace(/^\s*native_dirs\s*=\s*\{[^}\r\n]*\}\s*(?:#.*)?$/m, dependency =>
      /\bpackage\s*=\s*["']dirs["']/.test(dependency) ? '' : dependency);
    if (/\bdirs\b/.test(withoutNativeDependency)) {
      throw new Error('The dirs package must use only the reviewed native_dirs dependency; dirs belongs to the library compatibility boundary.');
    }
  }
  const targets = [sourceRoot, join(repositoryRoot, 'src-tauri', 'examples'), join(repositoryRoot, 'src-tauri', 'benches')];
  for (const path of targets.filter(existsSync).flatMap(rustFiles)) {
    if (path === join(sourceRoot, 'profile_paths.rs')) continue;
    const source = readFileSync(path, 'utf8');
    checkEnvironmentSyntax(source, path);
    let code = outsideProfileAdapter(rustCode(source));
    if (path === libraryRoot && compatibilityPrefix !== undefined) code = code.slice(compatibilityPrefix.length);
    checkDirectoryBindings(code, path);
    const libraryModule = path.startsWith(sourceRoot + sep);
    const separateTarget = path === join(sourceRoot, 'main.rs') || path.startsWith(join(sourceRoot, 'bin') + sep);
    if (compatibilityPrefix !== undefined && libraryModule && !separateTarget) code = withoutDirectoryCompatibility(code);
    if (/\bnative_dirs\b|\bdirs\s*::|\b(?:std\s*::\s*)?env\s*::\s*(?:var|var_os|vars|vars_os|set_var|remove_var|home_dir)\b|\buse\s+std\s*::\s*env\b|\b(?:allow|expect)\s*\([^)]*\bdisallowed_methods\b/.test(code)) {
      throw new Error(`Native profile/environment bypass outside profile_paths.rs: ${path}`);
    }
  }
  const externalTestRoots = ['tests', 'examples', 'benches'].map(name => join(repositoryRoot, 'src-tauri', name));
  for (const path of externalTestRoots.filter(existsSync).flatMap(rustFiles)) {
    const source = readFileSync(path, 'utf8');
    checkEnvironmentSyntax(source, path);
    const code = rustCode(source);
    for (const attribute of code.matchAll(/#\s*!?\s*\[\s*([\w:]+)/g)) {
      if (!['test', 'tokio::test', 'async_std::test', 'cfg', 'allow', 'ignore', 'serial', 'derive'].includes(attribute[1]!)) {
        throw new Error(`Unreviewed integration test attribute ${attribute[1]}: ${path}`);
      }
    }
    for (const statement of code.matchAll(/\buse\b[^;]*;/g)) {
      if (/\b(?:test|rstest|test_case)\b/.test(statement[0])) throw new Error(`Integration test macro aliases are unsupported: ${path}`);
    }
    for (const decorated of code.matchAll(/((?:#\s*\[[^\]]*\]\s*)+)(?:async\s+)?fn\s+\w+[^\{]*\{/g)) {
      if (/#\s*\[\s*serial\b/.test(decorated[1]!) && !/#\s*\[\s*(?:(?:tokio|async_std)\s*::\s*)?test\b/.test(decorated[1]!)) {
        throw new Error(`Integration serial attributes must accompany a recognized test: ${path}`);
      }
    }
    const tests = code.matchAll(/#\s*\[\s*(?:(?:tokio|async_std)\s*::\s*)?test\s*(?:\([^)]*\))?\s*\][\s\S]*?\bfn\s+(\w+)[^{]*\{/g);
    for (const match of tests) {
      const start = match.index + match[0].length;
      const first = code.slice(start);
      const live = path === join(repositoryRoot, 'src-tauri', 'tests', 'live_profile.rs');
      const prefix = live
        ? /^\s*assert!\s*\(\s*claude_code_history_viewer_lib\s*::\s*profile_paths\s*::\s*live_profile_tests_enabled\s*\(\s*\)/.exec(first)
        : /^\s*let\s+(?:_profile|profile)\s*=\s*claude_code_history_viewer_lib\s*::\s*profile_paths\s*::\s*TestProfile\s*::\s*new\s*\(\s*\)\s*\.\s*expect\s*\(/.exec(first);
      const tail = prefix ? source.slice(start + prefix[0].length) : '';
      const guarded = prefix !== null && (live
        ? /^(?:\s*|\s*,\s*"(?:\\.|[^"\\])*"\s*,?\s*)\)\s*;/.test(tail)
        : /^\s*"(?:\\.|[^"\\])*"\s*\)\s*;/.test(tail));
      if (!guarded) throw new Error(`Integration test ${match[1]} must acquire its profile guard before doing any work: ${path}`);
      if (live && !/#\s*\[\s*ignore\b/.test(match[0])) throw new Error(`Live test ${match[1]} must remain ignored by default.`);
    }
    if (/\bnative_dirs\b|\bdirs\s*::|\b(?:std\s*::\s*)?env\s*::\s*(?:var|var_os|vars|vars_os|set_var|remove_var|home_dir)\b/.test(outsideProfileAdapter(code))) {
      throw new Error(`Integration tests must use the shared profile boundary: ${path}`);
    }
  }
}

function main(): void {
  const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
  checkTestBoundary(repositoryRoot);
  const fixtureRoot = mkdtempSync(join(tmpdir(), 'cchv-test-profile-'));
  const inheritedEnv = { ...process.env };
  inheritedEnv['CARGO_HOME'] ??= join(homedir(), '.cargo');
  inheritedEnv['RUSTUP_HOME'] ??= join(homedir(), '.rustup');
  const invocation = buildInvocation(process.argv.slice(2), inheritedEnv, fixtureRoot, repositoryRoot);
  for (const name of ['home', 'config', 'data', 'cache', 'tmp']) mkdirSync(join(fixtureRoot, name));
  const receiptDirectory = join(repositoryRoot, 'scratch', 'test-profile-isolation');
  mkdirSync(receiptDirectory, { recursive: true });
  console.log(`[test-rust] ${invocation.command} ${invocation.args.join(' ')}`);
  console.log(invocation.env['CCHV_TEST_PROFILE'] === 'live-v1'
    ? '[test-rust] live-v1; native profile access acknowledged; provider caches can change'
    : `[test-rust] isolated-v1; profile: ${fixtureRoot}`);
  const result = spawnSync(invocation.command, invocation.args, { cwd: invocation.cwd, env: invocation.env, stdio: 'inherit', shell: false });
  const status = result.status ?? 1;
  const receipt = { command: invocation.command, args: invocation.args, profile: fixtureRoot, mode: invocation.env['CCHV_TEST_PROFILE'], target: invocation.env['CARGO_TARGET_DIR'], status, signal: result.signal, error: result.error?.message };
  writeFileSync(join(receiptDirectory, `${fixtureRoot.split(/[\\/]/).at(-1)}.json`), `${JSON.stringify(receipt, null, 2)}\n`);
  if (result.error) console.error(result.error.message);
  process.exitCode = status;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try { main(); }
  catch (error) { console.error(error instanceof Error ? error.message : error); process.exitCode = 1; }
}
