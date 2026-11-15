// Free-model aggregator: one server, two client wires, ten simulated free-tier providers.
//
// The server reads the trial brief (the provider directory, the quality table and the
// workload manifest), probes each provider's catalog route, builds one catalog, registers
// it with the env, and then answers the env client's twenty requests: each one with the
// best-rated free model that meets its requirements, failing over to the next candidate
// when a provider refuses. Answers are translated between the OpenAI, Anthropic and
// Gemini wires in both directions, in JSON and in server-sent events.
//
// Everything a request depends on comes from the brief, the request itself and the
// providers' answers: no wall clock and no unseeded randomness, so a replay leaves the
// same state export. Node 22 standard library only; run as `node main.ts`.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** A JSON object, the only shape this server passes around. */
type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** The protocol version the register body carries. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** How long one provider catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A fixed epoch second for every model listing and completion `created`: no clock in a reply. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every Anthropic listing's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief
// ---------------------------------------------------------------------------

interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (or absolute URL) the provider's routes hang off. */
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  manifest: ManifestRow[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/**
 * Every JSON object in `text` that carries a string `schema`, in the order they appear.
 * The brief's three documents are each one line, so a line scan finds them first; the
 * brace scanner below is the fallback for a brief that wraps them.
 */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{"schema"')) continue;
    try {
      const value: unknown = JSON.parse(trimmed);
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not one JSON document on its own line: the scanner below tries instead.
    }
  }
  if (out.length > 0) return out;
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function directoryProviderOf(v: unknown): DirectoryProvider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key: asString(v.key) };
}

function ratingRowOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestRowOf(v: unknown): ManifestRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const row: ManifestRow = {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {},
  };
  if (requires.tools === true) row.requires.tools = true;
  if (requires.vision === true) row.requires.vision = true;
  if (requires.json === true) row.requires.json = true;
  const minContext = asNumber(requires.min_context);
  if (minContext !== null) row.requires.min_context = minContext;
  return row;
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: 'the brief holds no free-model-aggregator-01/providers/v1 document' };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = directoryProviderOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingRowOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [{ id: asString(v.id), rating_milli: asNumber(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = manifestRowOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// The catalog probe
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** One model as a catalog listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context = asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit);
  return {
    id,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : id,
    free: typeof v.free === 'boolean' ? v.free : null,
    context_length: context,
    tools: typeof v.tools === 'boolean' ? v.tools : null,
    vision: typeof v.vision === 'boolean' ? v.vision : null,
    json: typeof v.json === 'boolean' ? v.json : null,
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: DirectoryProvider): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

function ratingRowFor(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table's row where it has one, the listing's own word otherwise. */
function modelIsFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface ModelCaps {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What a model can do: the quality table's row where it has one, the listing's fields otherwise. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): ModelCaps {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: ModelCaps): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every
 * model it listed. Its base_url is the directory's own path (`/p/<id>`), never the path
 * joined to the env URL: the env's origin differs between a live pass and its replay, and
 * the catalog is part of the state export a replay must reproduce byte for byte.
 */
function buildCatalog(probes: readonly Probe[], brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(brief, probe.provider.id, model.id);
      // Only free models go in the catalog: a paid model is never a legal pick, and
      // listing free models alone keeps this document identical whether it was built
      // from a probe or from the brief, so `--list` and the register body agree byte
      // for byte in every context. The catalog score reads the free set alone.
      if (!modelIsFree(model, row)) continue;
      models.push({ id: model.id, free: true, context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

interface Candidate {
  provider: DirectoryProvider;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** The providers whose free-tier budget is spent, or that are out for the trial. */
const deadProviders = new Set<string>();
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingRowFor(brief, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/**
 * Every free model that meets the requirements, best first: the quality table's rating,
 * then the provider id, then the model id, so the order never depends on arrival or on
 * the clock. Providers already known spent or out are left out.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (deadProviders.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingRowFor(brief, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, plus what its body shows. */
function requirementsFor(clientWire: Wire, body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(clientWire, body);
  const merged: Requirement = { ...fromBody };
  if (entry !== undefined) {
    if (entry.requires.tools === true) merged.tools = true;
    if (entry.requires.vision === true) merged.vision = true;
    if (entry.requires.json === true) merged.json = true;
    if (entry.requires.min_context !== undefined) merged.min_context = entry.requires.min_context;
  }
  return merged;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(clientWire: Wire, body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const json = clientWire === 'openai' && (format === 'json_object' || format === 'json_schema');
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json };
}

// ---------------------------------------------------------------------------
// Wire translation: client request -> provider request
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages request as an OpenAI Chat Completions request naming `model`. */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        const args = call.function.arguments;
        if (isRecord(args)) input = args;
        else if (typeof args === 'string' && args.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(args);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * A client request (OpenAI or Anthropic) as a Gemini generateContent request. Gemini has no
 * tool_choice: it is dropped, which the echo never reads. `stream` picks the streaming route.
 */
function toGemini(body: Json, clientWire: Wire): { url: string; payload: Json } | null {
  const contents: Json[] = [];
  let systemText = '';
  let tools: Json[] = [];
  let maxTokens: number | null = null;
  let stops: string[] = [];
  let jsonMode = false;
  let temperature: number | undefined;
  if (clientWire === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role !== 'user') continue;
      const parts = geminiPartsOfOpenAIContent(raw.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
    if (typeof body.stop === 'string') stops = [body.stop];
    else if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    jsonMode = format === 'json_object' || format === 'json_schema';
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (Array.isArray(body.tools)) tools = geminiToolsOfOpenAI(body.tools);
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role !== 'user') continue;
      const parts = geminiPartsOfAnthropicContent(raw.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    maxTokens = asNumber(body.max_tokens);
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (Array.isArray(body.tools)) tools = geminiToolsOfAnthropic(body.tools);
  }
  if (contents.length === 0) return null;
  const generationConfig: Json = {};
  if (maxTokens !== null) generationConfig.maxOutputTokens = maxTokens;
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (jsonMode) generationConfig.responseMimeType = 'application/json';
  if (temperature !== undefined) generationConfig.temperature = temperature;
  const payload: Json = { contents, generationConfig };
  if (systemText !== '') payload.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) payload.tools = [{ functionDeclarations: tools }];
  return { url: '', payload };
}

/** OpenAI message content as Gemini parts: text, or an image as inline data. */
function geminiPartsOfOpenAIContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(part.image_url.url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      else parts.push({ fileData: { fileUri: part.image_url.url } });
    }
  }
  return parts;
}

/** Anthropic message content as Gemini parts: text, or an image as inline data. */
function geminiPartsOfAnthropicContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    }
  }
  return parts;
}

function geminiToolsOfOpenAI(tools: unknown[]): Json[] {
  const out: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    out.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
  }
  return out;
}

function geminiToolsOfAnthropic(tools: unknown[]): Json[] {
  const out: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    out.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Provider calls
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop family: always the plain end, never length or tool use. */
  stop: 'stop' | 'end_turn';
  model: string;
  id: string;
  created: number;
}

interface CallOutcome {
  ok: boolean;
  answer?: ProviderAnswer;
  status?: number;
  code?: string;
  message?: string;
  retryAfter?: number;
}

/** The error envelope's code and message, in whichever wire the provider speaks. */
function errorOf(wire: Wire, body: unknown): { code: string; message: string; retryAfter: number | null } {
  if (!isRecord(body)) return { code: '', message: '', retryAfter: null };
  const error = isRecord(body.error) ? body.error : body;
  const retry = asNumber(error.retry_after);
  return { code: asString(error.code) !== '' ? asString(error.code) : asString(error.type), message: asString(error.message), retryAfter: retry };
}

/** Split a block of SSE text into its frames: each frame's event name (or null) and data lines. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** An OpenAI provider's answer, JSON or SSE, as one answer. */
function openAIAnswer(body: unknown, streamed: boolean): ProviderAnswer | null {
  if (streamed) {
    if (typeof body !== 'string') return null;
    let text = '';
    let input = 0;
    let output = 0;
    let id = 'chatcmpl-sim';
    let created = FIXED_CREATED;
    for (const frame of sseFrames(body)) {
      if (frame.data.trim() === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (typeof chunk.id === 'string') id = chunk.id;
      if (typeof chunk.created === 'number') created = chunk.created;
      if (isRecord(chunk.usage)) {
        input = asNumber(chunk.usage.prompt_tokens) ?? input;
        output = asNumber(chunk.usage.completion_tokens) ?? output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    }
    return { text, inputTokens: input, outputTokens: output, stop: 'stop', model: '', id, created };
  }
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: asNumber(usage.prompt_tokens) ?? 0,
    outputTokens: asNumber(usage.completion_tokens) ?? 0,
    stop: 'stop',
    model: asString(body.model),
    id: asString(body.id) !== '' ? asString(body.id) : 'chatcmpl-sim',
    created: asNumber(body.created) ?? FIXED_CREATED,
  };
}

/** An Anthropic provider's answer, JSON or SSE, as one answer. */
function anthropicAnswer(body: unknown, streamed: boolean): ProviderAnswer | null {
  if (streamed) {
    if (typeof body !== 'string') return null;
    let text = '';
    let input = 0;
    let output = 0;
    for (const frame of sseFrames(body)) {
      let data: unknown;
      try {
        data = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
        input = asNumber(data.message.usage.input_tokens) ?? input;
      } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        text += data.delta.text;
      } else if (frame.event === 'message_delta' && isRecord(data.usage)) {
        output = asNumber(data.usage.output_tokens) ?? output;
      }
    }
    return { text, inputTokens: input, outputTokens: output, stop: 'end_turn', model: '', id: 'msg_sim', created: FIXED_CREATED };
  }
  if (!isRecord(body)) return null;
  const text = textOf(body.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: asNumber(usage.input_tokens) ?? 0,
    outputTokens: asNumber(usage.output_tokens) ?? 0,
    stop: 'end_turn',
    model: asString(body.model),
    id: asString(body.id) !== '' ? asString(body.id) : 'msg_sim',
    created: FIXED_CREATED,
  };
}

/** A Gemini provider's answer, JSON or SSE, as one answer. */
function geminiAnswer(body: unknown, streamed: boolean): ProviderAnswer | null {
  if (streamed) {
    if (typeof body !== 'string') return null;
    let text = '';
    let input = 0;
    let output = 0;
    let model = '';
    for (const frame of sseFrames(body)) {
      let data: unknown;
      try {
        data = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (typeof data.modelVersion === 'string') model = data.modelVersion;
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      for (const cand of candidates) {
        if (!isRecord(cand) || !isRecord(cand.content) || !Array.isArray(cand.content.parts)) continue;
        for (const part of cand.content.parts) {
          if (isRecord(part) && typeof part.text === 'string') text += part.text;
        }
      }
      if (isRecord(data.usageMetadata)) {
        input = asNumber(data.usageMetadata.promptTokenCount) ?? input;
        output = asNumber(data.usageMetadata.candidatesTokenCount) ?? output;
      }
    }
    return { text, inputTokens: input, outputTokens: output, stop: 'stop', model, id: 'chatcmpl-sim', created: FIXED_CREATED };
  }
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  let text = '';
  if (first !== null && isRecord(first.content) && Array.isArray(first.content.parts)) {
    for (const part of first.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') text += part.text;
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: asNumber(usage.promptTokenCount) ?? 0,
    outputTokens: asNumber(usage.candidatesTokenCount) ?? 0,
    stop: 'stop',
    model: asString(body.modelVersion),
    id: 'chatcmpl-sim',
    created: FIXED_CREATED,
  };
}

/** One call to one provider on its own wire. A refusal is an outcome, never an exception. */
async function callProvider(candidate: Candidate, clientWire: Wire, body: Json, stream: boolean): Promise<CallOutcome> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) {
      headers.accept = 'text/event-stream';
      payload.stream_options = { include_usage: true };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const built = toGemini(body, clientWire);
    if (built === null) return { ok: false, status: 400, code: 'invalid_request', message: 'the request holds no content for the gemini wire' };
    url = stream
      ? `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:streamGenerateContent`
      : `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
    payload = built.payload;
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', message: (err as Error).message };
  }
  const retryHeader = res.headers.get('retry-after');
  const retryAfter = retryHeader !== null && retryHeader !== '' && Number.isFinite(Number(retryHeader)) ? Number(retryHeader) : undefined;
  if (!res.ok) {
    let parsed: unknown = null;
    try {
      parsed = await res.json();
    } catch {
      // The error body is not JSON: the status alone says it refused.
    }
    const err = errorOf(provider.wire, parsed);
    return { ok: false, status: res.status, code: err.code, message: err.message, retryAfter: retryAfter ?? err.retryAfter ?? undefined };
  }
  const parsed = stream ? await res.text() : await res.json();
  const answer =
    provider.wire === 'openai' ? openAIAnswer(parsed, stream) : provider.wire === 'anthropic' ? anthropicAnswer(parsed, stream) : geminiAnswer(parsed, stream);
  if (answer === null) return { ok: false, status: 502, code: 'bad_answer', message: `the ${provider.id} answer did not parse` };
  answer.model = candidate.model.id;
  return { ok: true, answer };
}

/** Wait `seconds`, at most five: a provider's retry-after, honoured before the next attempt. */
function waitSeconds(seconds: number): Promise<void> {
  const ms = Math.min(Math.max(seconds, 0), 5) * 1000;
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Serve one client request: try the candidates best first, and on a 402, a 429 or a
 * 5xx move to the next. A provider whose answer says its budget is spent, or that it is
 * out, is dead for the rest of the trial; a scheduled refusal is not. Exactly one
 * successful provider call answers a request: the first success ends the tries.
 */
async function serve(clientWire: Wire, body: Json, stream: boolean): Promise<ProviderAnswer | null> {
  const requires = requirementsFor(clientWire, body);
  const tried = new Set<string>();
  for (const candidate of candidatesFor(requires)) {
    const key = `${candidate.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const outcome = await callProvider(candidate, clientWire, body, stream);
    if (outcome.ok && outcome.answer !== undefined) return outcome.answer;
    if (outcome.status === 402 || outcome.status === 429 || (outcome.status !== undefined && outcome.status >= 500)) {
      const message = outcome.message ?? '';
      if (message.includes('budget is spent') || message.includes('is out')) deadProviders.add(candidate.provider.id);
      if (outcome.retryAfter !== undefined && outcome.retryAfter > 0) await waitSeconds(outcome.retryAfter);
      continue;
    }
    // A 400 or a 401 is this entry's own fault on that candidate: skip it and move on.
    log(`candidate ${key} refused with HTTP ${String(outcome.status ?? 0)} ${outcome.code ?? ''}: ${(outcome.message ?? '').slice(0, 200)}`);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Client responses
// ---------------------------------------------------------------------------

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text the way the providers split their streamed answers: 13 characters a piece. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A provider's answer as an OpenAI Chat Completions object. */
function answerToOpenAI(answer: ProviderAnswer): Json {
  return {
    id: answer.id,
    object: 'chat.completion',
    created: answer.created,
    model: answer.model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A provider's answer as an Anthropic Messages object. */
function answerToAnthropic(answer: ProviderAnswer): Json {
  return {
    id: `msg_${answer.id}`,
    type: 'message',
    role: 'assistant',
    model: answer.model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A provider's answer as an OpenAI event stream: role, content, finish, usage, [DONE]. */
function answerToOpenAIStream(answer: ProviderAnswer): string {
  const usage = { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens };
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: answer.id,
    object: 'chat.completion.chunk',
    created: answer.created,
    model: answer.model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(sseEvent(null, { id: answer.id, object: 'chat.completion.chunk', created: answer.created, model: answer.model, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A provider's answer as an Anthropic event stream, in the provider's own event order. */
function answerToAnthropicStream(answer: ProviderAnswer): string {
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: `msg_${answer.id}`, type: 'message', role: 'assistant', model: answer.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// The HTTP server
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, body: string, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendClientError(res: ServerResponse, wire: Wire, status: number, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(status === 503 ? 'overloaded_error' : 'api_error', message));
  else sendJson(res, status, openAIError(message, status === 503 ? 'service_unavailable' : 'invalid_request_error', null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, { data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })), has_more: false });
  } else {
    sendJson(res, 200, { object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) });
  }
}

/** POST /v1/chat/completions and POST /v1/messages: serve one request. */
async function serveRequest(req: IncomingMessage, res: ServerResponse, clientWire: Wire): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendClientError(res, clientWire, 400, 'the request body is not one JSON object');
    return;
  }
  const stream = body.stream === true;
  const answer = await serve(clientWire, body, stream);
  if (answer === null) {
    sendClientError(res, clientWire, 503, 'no free model on the directory could serve this request');
    return;
  }
  if (stream) {
    const text = clientWire === 'anthropic' ? answerToAnthropicStream(answer) : answerToOpenAIStream(answer);
    sendText(res, 200, text, 'text/event-stream; charset=utf-8');
    return;
  }
  sendJson(res, 200, clientWire === 'anthropic' ? answerToAnthropic(answer) : answerToOpenAI(answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveRequest(req, res, 'anthropic');
    return;
  }
  sendClientError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: ReturnType<typeof createServer>): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  // A provider whose listing did not answer still belongs in the catalog: the brief's
  // quality table names its models, their free flags and their context lengths, and the
  // listings agree with it by construction. Without this, `--list` outside a trial (no
  // env to probe) would print an empty catalog instead of the registered one.
  probes = probes.map((probe) => {
    if (probe.ok) return probe;
    const rows = brief.ratings.filter((r) => r.provider === probe.provider.id);
    if (rows.length === 0) return probe;
    return {
      ...probe,
      ok: true,
      models: rows.map((row) => ({ id: row.model, display_name: row.model, free: row.free, context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json })),
    };
  });
  catalog = buildCatalog(probes, brief);
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  for (let attempt = 1; attempt <= 20; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      if (attempt === 20) throw err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: ReturnType<typeof createServer>): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built
    // object and stop before any server starts. A process whose stdout is not flushed
    // loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendClientError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} probed providers and ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
