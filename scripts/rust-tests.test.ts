import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";

import { buildInvocation, checkTestBoundary } from "./test-rust.js";

const repositoryRoot = path.resolve("fixture repository with spaces");
const fixtureRoot = path.resolve("fixture profile with spaces");

function invoke(args: string[], inheritedEnv: NodeJS.ProcessEnv = {}) {
  return buildInvocation(args, inheritedEnv, fixtureRoot, repositoryRoot);
}

function environmentEntries(env: NodeJS.ProcessEnv, name: string) {
  return Object.entries(env).filter(([key]) => key.toUpperCase() === name);
}

test("the default invocation isolates cargo test and serializes libtest", () => {
  const inheritedEnv: NodeJS.ProcessEnv = {
    PATH: "fixture-toolchain-path",
    CARGO_HOME: "existing-cargo-cache",
    RUSTUP_HOME: "existing-rustup-cache",
  };
  const originalEnv = { ...inheritedEnv };

  const invocation = buildInvocation([], inheritedEnv, fixtureRoot, repositoryRoot);

  assert.equal(invocation.command, "cargo");
  assert.deepEqual(invocation.args, ["test", "--", "--test-threads=1"]);
  assert.equal(invocation.cwd, path.join(repositoryRoot, "src-tauri"));
  assert.equal(invocation.env.CCHV_TEST_PROFILE, "isolated-v1");
  assert.equal(
    invocation.env.CARGO_TARGET_DIR,
    path.join(repositoryRoot, "src-tauri", "target", "test-isolated"),
  );
  assert.equal(invocation.env.PATH, inheritedEnv.PATH);
  assert.equal(invocation.env.CARGO_HOME, inheritedEnv.CARGO_HOME);
  assert.equal(invocation.env.RUSTUP_HOME, inheritedEnv.RUSTUP_HOME);
  assert.deepEqual(inheritedEnv, originalEnv);
  assert.notEqual(invocation.env, inheritedEnv);
});

test("profile and temporary directories replace case-insensitive inherited aliases", () => {
  const expected: Record<string, string> = {
    HOME: path.join(fixtureRoot, "home"),
    USERPROFILE: path.join(fixtureRoot, "home"),
    APPDATA: path.join(fixtureRoot, "config"),
    LOCALAPPDATA: path.join(fixtureRoot, "data"),
    XDG_CONFIG_HOME: path.join(fixtureRoot, "config"),
    XDG_DATA_HOME: path.join(fixtureRoot, "data"),
    XDG_CACHE_HOME: path.join(fixtureRoot, "cache"),
    TEMP: path.join(fixtureRoot, "tmp"),
    TMP: path.join(fixtureRoot, "tmp"),
    TMPDIR: path.join(fixtureRoot, "tmp"),
    CARGO_TARGET_DIR: path.join(repositoryRoot, "src-tauri", "target", "test-isolated"),
    CCHV_TEST_PROFILE: "isolated-v1",
    RUST_TEST_THREADS: "1",
  };
  const inheritedEnv = Object.fromEntries(
    Object.keys(expected).flatMap((key) => [
      [key, "host-profile-a"],
      [key.toLowerCase(), "host-profile-b"],
    ]),
  );
  const originalEnv = { ...inheritedEnv };
  Object.freeze(inheritedEnv);

  const { env } = invoke([], inheritedEnv);

  for (const [key, value] of Object.entries(expected)) {
    assert.deepEqual(environmentEntries(env, key), [[key, value]], key);
  }
  assert.deepEqual(inheritedEnv, originalEnv);
});

test("provider and runner overrides are scrubbed case-insensitively", () => {
  const scrubbedNames = [
    "HOMEDRIVE", "HOMEPATH", "CODEX_HOME", "COPILOT_CLI_HOME", "CLAUDE_CONFIG_DIR",
    "CLAUDE_POWERPOINT_INDEXEDDB_PATH", "CONTINUE_GLOBAL_DIR", "FORGE_CONFIG",
    "GEMINI_HOME", "GOOSE_PATH_ROOT", "HERMES_HOME", "HERMES_DATA_DIR_SUFFIX",
    "INTERPRETER_HOME", "KIMI_SHARE_DIR", "KIMI_HOME", "LLM_USER_PATH", "OPENCODE_HOME",
    "QWEN_RUNTIME_DIR", "QWEN_HOME", "VIBE_HOME", "CCHV_TEST_VSCODE_USER_DATA_ROOT",
    "NEXTEST_PROFILE", "NEXTEST_USER_CONFIG_FILE", "NEXTEST_FUTURE_OVERRIDE",
    "CCHV_TOKEN", "FEEDBACK_EMAIL",
  ];
  const inheritedEnv: NodeJS.ProcessEnv = {
    PATH: "supported-toolchain-path",
    CARGO_HOME: "existing-cargo-cache",
    RUSTUP_HOME: "existing-rustup-cache",
  };
  for (const name of scrubbedNames) {
    inheritedEnv[name] = "host-override-a";
    inheritedEnv[name.toLowerCase()] = "host-override-b";
    inheritedEnv[`${name[0]}${name.slice(1).toLowerCase()}`] = "host-override-c";
  }
  const originalEnv = { ...inheritedEnv };
  Object.freeze(inheritedEnv);

  const { env } = invoke([], inheritedEnv);

  for (const name of scrubbedNames) {
    assert.deepEqual(environmentEntries(env, name), [], name);
  }
  assert.equal(env.PATH, inheritedEnv.PATH);
  assert.equal(env.CARGO_HOME, inheritedEnv.CARGO_HOME);
  assert.equal(env.RUSTUP_HOME, inheritedEnv.RUSTUP_HOME);
  assert.deepEqual(inheritedEnv, originalEnv);
});

test("ordinary warning denial survives without a caller-owned compiler configuration", () => {
  const inheritedEnv = { RUSTFLAGS: "-D warnings" };
  assert.equal(invoke([], inheritedEnv).env.RUSTFLAGS, "-D warnings");
  assert.equal(inheritedEnv.RUSTFLAGS, "-D warnings");
  assert.doesNotThrow(() => invoke([], { CARGO_ENCODED_RUSTFLAGS: "" }));
});

for (const rustflags of [
  "--cfg cc_history_test_isolation",
  "--cfg=cc_history_test_isolation",
  "--cfg not(cc_history_test_isolation)",
  "--config build.rustflags=[]",
  "-A warnings",
  "-C debuginfo=0",
]) {
  for (const key of ["RUSTFLAGS", "rustflags"]) {
    test(`caller compiler configuration is rejected: ${key}=${rustflags}`, () => {
      assert.throws(() => invoke([], { [key]: rustflags }));
    });
  }
}

for (const key of ["CARGO_ENCODED_RUSTFLAGS", "cargo_encoded_rustflags"]) {
  test(`encoded compiler configuration is rejected: ${key}`, () => {
    assert.throws(() => invoke([], { [key]: "--cfg\u001fcc_history_test_isolation" }));
    assert.throws(() => invoke([], { [key]: "-D\u001fwarnings" }));
  });
}

for (const name of [
  "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_TARGET",
  "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER", "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER",
]) {
  for (const key of [name, name.toLowerCase()]) {
    test(`inherited alternate build or execution surface is rejected: ${key}`, () => {
      assert.throws(() => invoke([], { [key]: "caller-owned-command-or-target" }));
      assert.doesNotThrow(() => invoke([], { [key]: "" }));
    });
  }
}

test("supported test options retain the filter as one literal argument", () => {
  const filter = "provider::case with spaces (a+b)";
  const args = ["test", "--features", "webui-server", "--filter", filter];
  Object.freeze(args);

  assert.deepEqual(invoke(args).args, [
    "test", "--features", "webui-server", filter, "--", "--test-threads=1",
  ]);
  assert.deepEqual(args, ["test", "--features", "webui-server", "--filter", filter]);
});

test("focused unit and named integration tests remain serial", () => {
  assert.deepEqual(invoke(["test", "--lib"]).args, [
    "test", "--lib", "--", "--test-threads=1",
  ]);
  assert.deepEqual(invoke(["test", "--test", "kimi_provider_test"]).args, [
    "test", "--test", "kimi_provider_test", "--", "--test-threads=1",
  ]);
});

test("nextest retains its process-per-test execution model", () => {
  assert.deepEqual(invoke(["nextest"]).args, ["nextest", "run"]);
  assert.deepEqual(invoke(["nextest", "--profile", "ci"]).args, [
    "nextest", "run", "--profile", "ci",
  ]);
});

for (const mode of ["nextest", "coverage"]) {
  test(`${mode} retains the name filter as a literal positional argument`, () => {
    const filter = "provider::case with spaces (a+b)";
    const { args } = invoke([mode, "--filter", filter]);
    assert.equal(args.at(-1), filter);
    assert.equal(args.filter((arg: string) => arg === filter).length, 1);
    assert.ok(!args.includes("--filter"));
    assert.ok(!args.includes("--test-threads=1"));
  });
}

test("coverage uses nextest and only its explicit open option", () => {
  assert.deepEqual(invoke(["coverage"]).args, ["llvm-cov", "nextest", "--html"]);
  assert.deepEqual(invoke(["coverage", "--open"]).args, [
    "llvm-cov", "nextest", "--html", "--open",
  ]);
});

test("coverage supports the existing Codecov LCOV output contract", () => {
  assert.deepEqual(invoke(["coverage", "--lcov"]).args, [
    "llvm-cov", "nextest", "--lcov", "--output-path", "lcov.info",
  ]);
});

test("clippy checks every target and feature with warnings denied", () => {
  assert.deepEqual(invoke(["clippy"]).args, [
    "clippy", "--all-targets", "--all-features", "--", "-D", "warnings",
  ]);
  assert.equal(invoke(["clippy"]).env.CCHV_TEST_PROFILE, "isolated-v1");
});

test("live diagnostics require the dedicated acknowledged invocation", () => {
  const inheritedEnv: NodeJS.ProcessEnv = {
    HOME: "native-home",
    USERPROFILE: "native-user-profile",
    CODEX_HOME: "native-codex-root",
    CCHV_TEST_PROFILE: "caller-profile",
    CARGO_TARGET_DIR: "caller-target",
  };
  const originalEnv = { ...inheritedEnv };
  Object.freeze(inheritedEnv);

  const invocation = invoke(["live", "--acknowledge-profile-access"], inheritedEnv);

  assert.equal(invocation.command, "cargo");
  assert.deepEqual(invocation.args, [
    "test", "--test", "live_profile", "--", "--ignored", "--test-threads=1",
  ]);
  assert.equal(invocation.env.CCHV_TEST_PROFILE, "live-v1");
  assert.equal(
    invocation.env.CARGO_TARGET_DIR,
    path.join(repositoryRoot, "src-tauri", "target", "test-live"),
  );
  assert.equal(invocation.env.HOME, inheritedEnv.HOME);
  assert.equal(invocation.env.USERPROFILE, inheritedEnv.USERPROFILE);
  assert.equal(invocation.env.CODEX_HOME, inheritedEnv.CODEX_HOME);
  assert.deepEqual(inheritedEnv, originalEnv);
});

for (const mode of ["test", "nextest", "coverage"]) {
  test(`${mode} accepts the reviewed webui-server feature`, () => {
    const { args } = invoke([mode, "--features", "webui-server"]);
    const featureIndex = args.indexOf("--features");
    assert.notEqual(featureIndex, -1);
    assert.equal(args[featureIndex + 1], "webui-server");
  });

  test(`${mode} accepts the explicit all-features switch`, () => {
    assert.ok(invoke([mode, "--all-features"]).args.includes("--all-features"));
  });
}

for (const mode of ["nextest", "coverage"]) {
  for (const profile of ["ci", "default", "quick", "coverage"]) {
    test(`${mode} accepts the reviewed ${profile} profile`, () => {
      const { args } = invoke([mode, "--profile", profile]);
      const profileIndex = args.indexOf("--profile");
      assert.notEqual(profileIndex, -1);
      assert.equal(args[profileIndex + 1], profile);
    });
  }
}

const rejectedArgs: string[][] = [
  ["unknown-mode"],
  ["--", "--ignored"],
  ["test", "--ignored"],
  ["test", "--include-ignored"],
  ["test", "--test-threads", "8"],
  ["test", "--test-threads=8"],
  ["test", "--manifest-path", "other/Cargo.toml"],
  ["test", "--manifest-path=other/Cargo.toml"],
  ["test", "--target-dir", "host-target"],
  ["test", "--target", "other-target"],
  ["test", "--config", "build.target-dir='host-target'"],
  ["test", "--release"],
  ["test", "--features", "unreviewed-feature"],
  ["test", "--features", "webui-server,unreviewed-feature"],
  ["test", "--features"],
  ["test", "--filter"],
  ["test", "--filter", ""],
  ["test", "--filter", "--ignored"],
  ["test", "--profile", "ci"],
  ["test", "--test", "live_profile"],
  ["test", "--test", "../live_profile"],
  ["test", "--test", "--ignored"],
  ["test", "--open"],
  ["test", "--lcov"],
  ["nextest", "--ignored"],
  ["nextest", "--run-ignored", "all"],
  ["nextest", "--profile", "unreviewed-profile"],
  ["nextest", "--test", "live_profile"],
  ["nextest", "--open"],
  ["nextest", "--lcov"],
  ["coverage", "--target-dir", "host-target"],
  ["coverage", "--lcov", "--open"],
  ["coverage", "--open", "--lcov"],
  ["clippy", "--", "-A", "warnings"],
  ["clippy", "--filter", "case"],
  ["clippy", "--lcov"],
  ["live"],
  ["live", "--acknowledge-profile-access", "--ignored"],
  ["test", "--acknowledge-profile-access"],
  ["--live-profile"],
];

for (const args of rejectedArgs) {
  test(`unsafe or unsupported arguments fail closed: ${JSON.stringify(args)}`, () => {
    assert.throws(() => invoke(args));
  });
}

function boundaryFixture(context: TestContext) {
  const temporaryParent = path.resolve(tmpdir());
  const root = mkdtempSync(path.join(temporaryParent, "cchv-runner-contract-"));
  mkdirSync(path.join(root, "src-tauri", "src"), { recursive: true });
  mkdirSync(path.join(root, "src-tauri", "tests"), { recursive: true });
  context.after(() => {
    assert.equal(path.dirname(root), temporaryParent);
    assert.ok(path.basename(root).startsWith("cchv-runner-contract-"));
    rmSync(root, { recursive: true, force: true });
  });
  return {
    root,
    write(relativePath: string, source: string) {
      const target = path.join(root, "src-tauri", relativePath);
      mkdirSync(path.dirname(target), { recursive: true });
      writeFileSync(target, source);
    },
  };
}

const firstProfileGuard = 'let _profile = claude_code_history_viewer_lib::profile_paths::TestProfile::new().expect("fixture profile");';
const firstLiveAssertion = "assert!(claude_code_history_viewer_lib::profile_paths::live_profile_tests_enabled());";

test("source preflight permits the shared native adapter and an exact first fixture guard", (context) => {
  const fixture = boundaryFixture(context);
  fixture.write("src/profile_paths.rs", `
    use std::env;
    fn native_profile() {
      let _ = native_dirs::home_dir();
      let _ = env::var("HOME");
    }
  `);
  fixture.write("src/lib.rs", "pub mod profile_paths;");
  fixture.write("tests/fixture.rs", `
    #[test]
    fn fixture_test() {
      ${firstProfileGuard}
      let _ = 1;
    }
  `);

  assert.doesNotThrow(() => checkTestBoundary(fixture.root));
});

for (const call of [
  'var("HOME")', 'var_os("HOME")', 'set_var("HOME", "fixture")', 'remove_var("HOME")',
]) {
  test(`source preflight permits the fully qualified profile environment helper ${call}`, (context) => {
    const fixture = boundaryFixture(context);
    fixture.write("src/lib.rs", `
      fn scoped_profile() { let _ = crate::profile_paths::env::${call}; }
    `);
    fixture.write("tests/fixture.rs", `
      #[test]
      fn fixture_test() {
        ${firstProfileGuard}
        let _ = claude_code_history_viewer_lib::profile_paths::env::${call};
      }
    `);

    assert.doesNotThrow(() => checkTestBoundary(fixture.root));
  });
}

test("source preflight rejects raw integration environment access after a valid profile guard", (context) => {
  const fixture = boundaryFixture(context);
  fixture.write("tests/fixture.rs", `
    #[test]
    fn fixture_test() {
      ${firstProfileGuard}
      let _ = claude_code_history_viewer_lib::profile_paths::env::var("HOME");
      let _ = std::env::var("HOME");
    }
  `);

  assert.throws(() => checkTestBoundary(fixture.root));
});

for (const [name, source] of [
  ["native directory calls", "fn bypass() { let _ = native_dirs::home_dir(); }"],
  ["legacy directory imports", "fn bypass() { let _ = dirs::home_dir(); }"],
  ["raw standard environment calls", 'fn bypass() { let _ = std::env::var("HOME"); }'],
  ["raw standard environment enumeration", 'fn bypass() { let _ = std::env::vars(); }'],
  ["raw standard OS environment enumeration", 'fn bypass() { let _ = std::env::vars_os(); }'],
  ["raw environment aliases", 'use std::env; fn bypass() { let _ = env::var_os("HOME"); }'],
  ["grouped standard environment aliases", 'use std::{env as ambient}; fn bypass() { let _ = ambient::var_os("HOME"); }'],
  ["compile-time home lookup", 'const HOME: &str = env!("HOME");'],
  ["optional compile-time provider lookup", 'const CODEX: Option<&str> = option_env!("CODEX_HOME");'],
  ["computed compile-time environment key", 'const HOME: &str = env!(concat!("HO", "ME"));'],
  ["local lint allowance", "#[allow(clippy::disallowed_methods)] fn bypass() {}"],
  ["local lint expectation", "#[expect(clippy::disallowed_methods)] fn bypass() {}"],
]) {
  test(`source preflight rejects ${name} outside the native adapter`, (context) => {
    const fixture = boundaryFixture(context);
    fixture.write("src/commands/nested.rs", source);

    assert.throws(() => checkTestBoundary(fixture.root));
  });
}

test("source preflight permits only the reviewed compile-time Cargo metadata", (context) => {
  const fixture = boundaryFixture(context);
  fixture.write("src/lib.rs", 'const VERSION: &str = env!("CARGO_PKG_VERSION");');
  fixture.write("tests/fixture.rs", `#[test] fn fixture_test() { ${firstProfileGuard} let _ = env!("CARGO_MANIFEST_DIR"); }`);
  assert.doesNotThrow(() => checkTestBoundary(fixture.root));
});

for (const source of [
  '#[rstest] fn fixture_test() {}',
  '#[custom_test] fn fixture_test() {}',
  'use tokio::test as test_case; #[test_case] async fn fixture_test() {}',
  'use tokio::{test as serial}; #[serial] async fn fixture_test() {}',
  '#[serial] fn fixture_test() {}',
]) {
  test(`source preflight rejects unreviewed test macros: ${source}`, (context) => {
    const fixture = boundaryFixture(context);
    fixture.write("tests/fixture.rs", source);
    assert.throws(() => checkTestBoundary(fixture.root));
  });
}

for (const key of ['CARGO_ALIAS_NEXTEST', 'cargo_alias_llvm_cov', 'Cargo_Alias_Clippy']) {
  test(`runner rejects Cargo alias overrides: ${key}`, () => {
    assert.throws(() => invoke(['nextest'], { [key]: 'unsafe-command' }));
  });
}

for (const [name, body] of [
  ["missing fixture guard", "let _ = 1;"],
  ["fixture guard after the first operation", `let _ = 1; ${firstProfileGuard}`],
  ["fixture guard with an executable failure message", 'let _profile = claude_code_history_viewer_lib::profile_paths::TestProfile::new().expect({ profile_write(); "message" });'],
]) {
  test(`source preflight rejects an integration test with ${name}`, (context) => {
    const fixture = boundaryFixture(context);
    fixture.write("tests/fixture.rs", `#[test] fn fixture_test() { ${body} }`);

    assert.throws(() => checkTestBoundary(fixture.root));
  });
}

test("source preflight permits ignored live tests with the first live assertion", (context) => {
  const fixture = boundaryFixture(context);
  fixture.write("tests/live_profile.rs", `
    #[test]
    #[ignore]
    fn live_diagnostic() {
      ${firstLiveAssertion}
      let _ = 1;
    }
  `);

  assert.doesNotThrow(() => checkTestBoundary(fixture.root));
});

for (const [name, attributes, body] of [
  ["missing ignore", "#[test]", firstLiveAssertion],
  ["missing first live assertion", "#[test] #[ignore]", "let _ = 1;"],
  ["live assertion after the first operation", "#[test] #[ignore]", `let _ = 1; ${firstLiveAssertion}`],
  ["unrelated first assertion", "#[test] #[ignore]", "assert!(true);"],
  ["always-true live assertion", "#[test] #[ignore]", "assert!(claude_code_history_viewer_lib::profile_paths::live_profile_tests_enabled() || true);"],
  ["inverted live assertion", "#[test] #[ignore]", "assert!(claude_code_history_viewer_lib::profile_paths::live_profile_tests_enabled() == false);"],
  ["live assertion with arguments", "#[test] #[ignore]", "assert!(claude_code_history_viewer_lib::profile_paths::live_profile_tests_enabled(true));"],
  ["live assertion with executable format arguments", "#[test] #[ignore]", 'assert!(claude_code_history_viewer_lib::profile_paths::live_profile_tests_enabled(), "{}", profile_write());'],
]) {
  test(`source preflight rejects a live diagnostic with ${name}`, (context) => {
    const fixture = boundaryFixture(context);
    fixture.write("tests/live_profile.rs", `${attributes} fn live_diagnostic() { ${body} }`);

    assert.throws(() => checkTestBoundary(fixture.root));
  });
}

test("source preflight ignores bypass-looking tokens only inside literals and comments", (context) => {
  const fixture = boundaryFixture(context);
  fixture.write("src/lib.rs", String.raw`
    // native_dirs::home_dir(); std::env::var("HOME");
    /* #[allow(clippy::disallowed_methods)] use std::env; */
    const TEXT: &str = "native_dirs::home_dir(); std::env::var(\"HOME\");";
    const RAW: &str = r##"use std::env; #[expect(clippy::disallowed_methods)]"##;
  `);
  fixture.write("tests/fixture.rs", `
    #[test]
    fn fixture_test() {
      // native_dirs::home_dir(); env::set_var("HOME", "host");
      ${firstProfileGuard}
      let _ = "native_dirs::home_dir(); std::env::var(HOME)";
    }
  `);

  assert.doesNotThrow(() => checkTestBoundary(fixture.root));
});

test("source preflight cannot hide unguarded tests behind quote character literals", (context) => {
  const fixture = boundaryFixture(context);
  fixture.write("tests/fixture.rs", `const QUOTE: char = '"'; #[test] fn unguarded() { public_writer("native"); }`);
  assert.throws(() => checkTestBoundary(fixture.root));
});

for (const target of ['examples', 'benches']) {
  test(`source preflight covers native lookups and unguarded tests in ${target}`, (context) => {
    const fixture = boundaryFixture(context);
    fixture.write(`${target}/probe.rs`, 'fn main() { let _ = native_dirs::home_dir(); }');
    assert.throws(() => checkTestBoundary(fixture.root));
    fixture.write(`${target}/probe.rs`, '#[test] fn unguarded() { public_writer(); }');
    assert.throws(() => checkTestBoundary(fixture.root));
  });
}

test("source preflight handles nested comments, character escapes, raw strings and lifetimes", (context) => {
  const fixture = boundaryFixture(context);
  fixture.write("tests/fixture.rs", String.raw`
    /* outer /* nested */ " ignored */
    const QUOTE: char = '"';
    const ESCAPE: char = '\'';
    const BYTE: u8 = b'"';
    const UNICODE: char = '\u{22}';
    const RAW: &str = r##" fake #[test] fn hidden() {} "##;
    fn borrowed<'a>(value: &'a str) -> &'a str { value }
    #[test] fn guarded() {
      let _profile = claude_code_history_viewer_lib::profile_paths::TestProfile::new().expect("fixture");
    }
  `);
  assert.doesNotThrow(() => checkTestBoundary(fixture.root));
});
