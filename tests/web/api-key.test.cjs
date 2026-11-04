// Repository LICENSE. Exercise the actual page functions with browser storage failures.
"use strict";
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

const source = readFileSync(path.join(__dirname, "../../serve/web/app.js"), "utf8");
function between(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "page section is present");
  return source.slice(a, b);
}
const script = between("const store = {", "// ------------------------------------------------------------------ toasts") +
  between("// ------------------------------------------------------------------ server access", "let health =");

function page({initial = "", readBlocked = false, writeBlocked = false} = {}) {
  const saved = new Map([["strata.apikey", JSON.stringify(initial)]]);
  const input = {value: ""}, notices = [];
  const context = vm.createContext({
    $: (id) => { assert.equal(id, "api-key"); return input; },
    toast: (...args) => notices.push(args),
    localStorage: {
      getItem(key) { if (readBlocked) throw Error("storage blocked"); return saved.get(key) ?? null; },
      setItem(key, value) { if (writeBlocked) throw Error("storage full"); saved.set(key, value); },
    },
  });
  vm.runInContext(script, context, {timeout: 1000});
  const headers = (json = false) => JSON.parse(JSON.stringify(context.headers(json)));
  return {input, notices, saved, headers};
}

test("an edited key takes effect before blur and persists when storage works", () => {
  const p = page({initial: "old-key"});
  assert.equal(p.input.value, "old-key");
  assert.deepEqual(p.headers(), {Authorization: "Bearer old-key"});
  p.input.value = "  new-key  ";
  p.input.oninput();
  assert.deepEqual(p.headers(true), {Authorization: "Bearer new-key", "Content-Type": "application/json"});
  assert.equal(JSON.parse(p.saved.get("strata.apikey")), "new-key");
  assert.equal(p.notices.length, 0);
  p.input.onchange();
  assert.equal(p.notices[0][0], "success");
});

test("blocked storage still uses the current key and reports page-only persistence", () => {
  const p = page({readBlocked: true, writeBlocked: true});
  assert.deepEqual(p.headers(), {});
  p.input.value = "page-key";
  p.input.oninput();
  assert.deepEqual(p.headers(), {Authorization: "Bearer page-key"});
  assert.equal(p.notices.length, 0);
  p.input.onchange();
  assert.equal(p.notices[0][0], "warn");
  assert.match(p.notices[0][2], /after reloading/);
});

test("a failed write never restores the previous stored key", () => {
  const p = page({initial: "stale-key", writeBlocked: true});
  p.input.value = "replacement-key";
  p.input.oninput();
  assert.deepEqual(p.headers(), {Authorization: "Bearer replacement-key"});
  assert.equal(JSON.parse(p.saved.get("strata.apikey")), "stale-key");
  p.input.onchange();
  assert.equal(p.notices[0][0], "warn");
});

test("clearing a key removes Authorization even when persistence fails", () => {
  for (const writeBlocked of [false, true]) {
    const p = page({initial: "old-key", writeBlocked});
    p.input.value = "   ";
    p.input.oninput();
    assert.deepEqual(p.headers(), {});
    assert.deepEqual(p.headers(true), {"Content-Type": "application/json"});
    p.input.onchange();
    assert.equal(p.notices[0][0], writeBlocked ? "warn" : "success");
  }
});
