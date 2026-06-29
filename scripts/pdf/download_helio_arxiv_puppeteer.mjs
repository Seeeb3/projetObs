#!/usr/bin/env node

/**
 * Download helio arXiv PDFs through headless Chrome and stealth mode.
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const SCRIPT_URL = import.meta.url;
const SCRIPT_PATH = fileURLToPath(SCRIPT_URL);
const PROJECT_ROOT = path.resolve(path.dirname(SCRIPT_PATH), "..", "..");
const DEFAULT_INPUT_CSV = path.join(
  PROJECT_ROOT,
  "data/processed/results/WIESP2022-NER_all_helio_only_merged.csv",
);
const DEFAULT_OUTPUT_DIR = path.join(PROJECT_ROOT, "data/raw/arxiv/helio");
const DEFAULT_AUDIT_CSV = path.join(
  PROJECT_ROOT,
  "artifacts/logs/helio_arxiv_puppeteer_audit_2026-05-29.csv",
);
const DEFAULT_TARGET_COUNT = 53;
const DEFAULT_CHROME_EXECUTABLE =
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const DEFAULT_ABS_TIMEOUT_MS = 20_000;
const DEFAULT_PDF_TIMEOUT_MS = 30_000;
const DEFAULT_PASS_SLEEP_MS = 60_000;
const DEFAULT_PASS_RETRY_COUNT = 3;
const DEFAULT_BROWSER_RESET_INTERVAL = 10;
const DEFAULT_BROWSER_FAILURE_THRESHOLD = 2;
const MIN_JITTER_MS = 2_000;
const MAX_JITTER_MS = 5_000;
const PDF_CONTENT_TYPE = "application/pdf";
const ARXIV_ID_SEPARATOR = "|";

const STATUS = Object.freeze({
  ALREADY_PRESENT: "already_present",
  DOWNLOADED: "downloaded",
  ABS_TIMEOUT: "abs_timeout",
  PDF_TIMEOUT: "pdf_timeout",
  ABS_HTTP_ERROR: "abs_http_error",
  PDF_HTTP_ERROR: "pdf_http_error",
  PDF_RESPONSE_MISSING: "pdf_response_missing",
  NON_PDF_RESPONSE: "non_pdf_response",
  WRITE_FAILED: "write_failed",
});

const CSV_HEADERS = Object.freeze([
  "bibcode",
  "title",
  "arxiv_id",
  "status",
  "attempt_count",
  "pdf_found",
  "pdf_path",
  "abs_url",
  "pdf_url",
  "last_http_status",
  "last_content_type",
  "last_error",
  "last_attempt_at",
  "downloaded_at",
]);

/**
 * @typedef {object} DownloadRuntime
 * @property {import("puppeteer-extra").PuppeteerExtra} puppeteer
 * @property {(input: string, options: object) => Array<Record<string, string>>} csvParse
 * @property {(input: Array<Record<string, unknown>>, options: object) => string} csvStringify
 */

/**
 * @typedef {object} CliOptions
 * @property {string} inputCsv
 * @property {string} outputDir
 * @property {string} auditCsv
 * @property {number} targetCount
 * @property {string} chromeExecutable
 * @property {number} absTimeoutMs
 * @property {number} pdfTimeoutMs
 * @property {number} limit
 */

/**
 * @typedef {object} DownloadRow
 * @property {string} bibcode
 * @property {string} title
 * @property {string} arxivId
 */

/**
 * @typedef {object} AuditRow
 * @property {string} bibcode
 * @property {string} title
 * @property {string} arxiv_id
 * @property {string} status
 * @property {number} attempt_count
 * @property {boolean} pdf_found
 * @property {string} pdf_path
 * @property {string} abs_url
 * @property {string} pdf_url
 * @property {string} last_http_status
 * @property {string} last_content_type
 * @property {string} last_error
 * @property {string} last_attempt_at
 * @property {string} downloaded_at
 */

/**
 * Parse one pipe-delimited arXiv CSV cell.
 *
 * @param {string | undefined | null} rawArxivIds
 * @returns {string[]}
 */
export function parseArxivIdsCell(rawArxivIds) {
  if (rawArxivIds === undefined || rawArxivIds === null) {
    return [];
  }

  /** @type {string[]} */
  const orderedIds = [];
  const seenIds = new Set();
  for (const rawValue of String(rawArxivIds).split(ARXIV_ID_SEPARATOR)) {
    const cleanedValue = rawValue.trim();
    if (!cleanedValue || seenIds.has(cleanedValue)) {
      continue;
    }
    seenIds.add(cleanedValue);
    orderedIds.push(cleanedValue);
  }
  return orderedIds;
}

/**
 * Sanitize an arXiv identifier for directory naming.
 *
 * @param {string} arxivId
 * @returns {string}
 */
export function sanitizeArxivIdForPath(arxivId) {
  return arxivId.trim().replaceAll("/", "__");
}

/**
 * Return an ISO timestamp.
 *
 * @returns {string}
 */
export function nowIso() {
  return new Date().toISOString();
}

/**
 * Escape one value for CSV output.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function csvEscape(value) {
  const text = value === undefined || value === null ? "" : String(value);
  return `"${text.replaceAll(`"`, `""`)}"`;
}

/**
 * Return a deterministic CSV fallback string.
 *
 * @param {AuditRow[]} rows
 * @returns {string}
 */
export function buildAuditCsvFallback(rows) {
  const lines = [CSV_HEADERS.join(",")];
  for (const row of rows) {
    lines.push(CSV_HEADERS.map((header) => csvEscape(row[header])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

/**
 * Sleep for the specified duration.
 *
 * @param {number} milliseconds
 * @returns {Promise<void>}
 */
export async function sleep(milliseconds) {
  await new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

/**
 * Return a jitter interval in milliseconds.
 *
 * @param {number} minMilliseconds
 * @param {number} maxMilliseconds
 * @returns {number}
 */
export function randomJitterMilliseconds(
  minMilliseconds = MIN_JITTER_MS,
  maxMilliseconds = MAX_JITTER_MS,
) {
  if (maxMilliseconds <= minMilliseconds) {
    return minMilliseconds;
  }

  const span = maxMilliseconds - minMilliseconds + 1;
  return minMilliseconds + Math.floor(Math.random() * span);
}

/**
 * Build the default bootstrap command for this script.
 *
 * @returns {string}
 */
export function buildBootstrapCommand() {
  return [
    "npm exec --yes",
    "--package=puppeteer-core",
    "--package=puppeteer-extra",
    "--package=puppeteer-extra-plugin-stealth",
    "--package=csv",
    "--",
    "node",
    JSON.stringify(SCRIPT_PATH),
    "--",
  ].join(" ");
}

/**
 * Parse CLI options.
 *
 * @param {string[]} argv
 * @returns {CliOptions}
 */
export function parseCliOptions(argv) {
  const parsed = parseArgs({
    args: argv,
    options: {
      "input-csv": { type: "string", default: DEFAULT_INPUT_CSV },
      "output-dir": { type: "string", default: DEFAULT_OUTPUT_DIR },
      "audit-csv": { type: "string", default: DEFAULT_AUDIT_CSV },
      "target-count": { type: "string", default: String(DEFAULT_TARGET_COUNT) },
      "chrome-executable": {
        type: "string",
        default: DEFAULT_CHROME_EXECUTABLE,
      },
      "abs-timeout-ms": {
        type: "string",
        default: String(DEFAULT_ABS_TIMEOUT_MS),
      },
      "pdf-timeout-ms": {
        type: "string",
        default: String(DEFAULT_PDF_TIMEOUT_MS),
      },
      limit: { type: "string", default: "0" },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: false,
  });

  if (parsed.values.help) {
    printHelp();
    process.exit(0);
  }

  const targetCount = Number(parsed.values["target-count"]);
  const absTimeoutMs = Number(parsed.values["abs-timeout-ms"]);
  const pdfTimeoutMs = Number(parsed.values["pdf-timeout-ms"]);
  const limit = Number(parsed.values.limit);
  if (!Number.isInteger(targetCount) || targetCount < 1) {
    throw new Error("--target-count must be a positive integer.");
  }
  if (!Number.isFinite(absTimeoutMs) || absTimeoutMs <= 0) {
    throw new Error("--abs-timeout-ms must be a positive number.");
  }
  if (!Number.isFinite(pdfTimeoutMs) || pdfTimeoutMs <= 0) {
    throw new Error("--pdf-timeout-ms must be a positive number.");
  }
  if (!Number.isInteger(limit) || limit < 0) {
    throw new Error("--limit must be a non-negative integer.");
  }

  return {
    inputCsv: path.resolve(String(parsed.values["input-csv"])),
    outputDir: path.resolve(String(parsed.values["output-dir"])),
    auditCsv: path.resolve(String(parsed.values["audit-csv"])),
    targetCount,
    chromeExecutable: String(parsed.values["chrome-executable"]),
    absTimeoutMs,
    pdfTimeoutMs,
    limit,
  };
}

/**
 * Print CLI usage.
 */
export function printHelp() {
  const lines = [
    "Usage: download_helio_arxiv_puppeteer.mjs [options]",
    "",
    "Options:",
    `  --input-csv PATH          Default: ${DEFAULT_INPUT_CSV}`,
    `  --output-dir PATH         Default: ${DEFAULT_OUTPUT_DIR}`,
    `  --audit-csv PATH          Default: ${DEFAULT_AUDIT_CSV}`,
    `  --target-count INTEGER    Default: ${DEFAULT_TARGET_COUNT}`,
    `  --chrome-executable PATH  Default: ${DEFAULT_CHROME_EXECUTABLE}`,
    `  --abs-timeout-ms INTEGER  Default: ${DEFAULT_ABS_TIMEOUT_MS}`,
    `  --pdf-timeout-ms INTEGER  Default: ${DEFAULT_PDF_TIMEOUT_MS}`,
    "  --limit INTEGER           Default: 0",
    "  --help, -h                Show this help text",
    "",
    "If dependencies are not installed locally, run this script through npm exec.",
  ];
  process.stdout.write(`${lines.join("\n")}\n`);
}

/**
 * Read rows from the input CSV.
 *
 * @param {string} inputCsv
 * @param {(input: string, options: object) => Array<Record<string, string>>} csvParse
 * @param {number} limit
 * @returns {Promise<DownloadRow[]>}
 */
export async function loadDownloadRows(inputCsv, csvParse, limit) {
  const csvText = await fsp.readFile(inputCsv, "utf8");
  const parsedRows = csvParse(csvText, {
    columns: true,
    skip_empty_lines: true,
  });

  /** @type {DownloadRow[]} */
  const downloadRows = [];
  const seenIds = new Set();
  for (const row of parsedRows) {
    const bibcode = String(row.bibcode ?? "").trim();
    const title = String(row.title ?? "").trim();
    for (const arxivId of parseArxivIdsCell(row.arxiv_ids)) {
      if (!bibcode || !title || seenIds.has(arxivId)) {
        continue;
      }
      seenIds.add(arxivId);
      downloadRows.push({ bibcode, title, arxivId });
      if (limit > 0 && downloadRows.length >= limit) {
        return downloadRows;
      }
    }
  }

  return downloadRows;
}

/**
 * Build the per-ID output directory.
 *
 * @param {string} outputDir
 * @param {string} arxivId
 * @returns {string}
 */
export function buildArxivOutputDirectory(outputDir, arxivId) {
  return path.join(outputDir, sanitizeArxivIdForPath(arxivId));
}

/**
 * Return sorted PDF paths within a directory.
 *
 * @param {string} targetDirectory
 * @returns {Promise<string[]>}
 */
export async function listPdfPaths(targetDirectory) {
  try {
    const directoryEntries = await fsp.readdir(targetDirectory, {
      withFileTypes: true,
    });
    return directoryEntries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".pdf"))
      .map((entry) => path.join(targetDirectory, entry.name))
      .sort();
  } catch (error) {
    if (/** @type {NodeJS.ErrnoException} */ (error).code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

/**
 * Find the first PDF within an output directory.
 *
 * @param {string} targetDirectory
 * @returns {Promise<string>}
 */
export async function findExistingPdf(targetDirectory) {
  const pdfPaths = await listPdfPaths(targetDirectory);
  return pdfPaths[0] ?? "";
}

/**
 * Count actual PDFs on disk.
 *
 * @param {string} outputDir
 * @returns {Promise<number>}
 */
export async function countActualPdfFiles(outputDir) {
  let count = 0;
  let directoryEntries = [];
  try {
    directoryEntries = await fsp.readdir(outputDir, { withFileTypes: true });
  } catch (error) {
    if (/** @type {NodeJS.ErrnoException} */ (error).code === "ENOENT") {
      return 0;
    }
    throw error;
  }

  for (const entry of directoryEntries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const targetDirectory = path.join(outputDir, entry.name);
    const pdfPaths = await listPdfPaths(targetDirectory);
    count += pdfPaths.length;
  }
  return count;
}

/**
 * Create an empty audit row.
 *
 * @param {DownloadRow} row
 * @returns {AuditRow}
 */
export function createAuditRow(row) {
  return {
    bibcode: row.bibcode,
    title: row.title,
    arxiv_id: row.arxivId,
    status: "",
    attempt_count: 0,
    pdf_found: false,
    pdf_path: "",
    abs_url: `https://arxiv.org/abs/${row.arxivId}`,
    pdf_url: "",
    last_http_status: "",
    last_content_type: "",
    last_error: "",
    last_attempt_at: "",
    downloaded_at: "",
  };
}

/**
 * Write the current audit CSV.
 *
 * @param {Map<string, AuditRow>} auditRowsById
 * @param {string} auditCsv
 * @param {(input: Array<Record<string, unknown>>, options: object) => string} csvStringify
 * @returns {Promise<void>}
 */
export async function writeAuditCsv(auditRowsById, auditCsv, csvStringify) {
  await fsp.mkdir(path.dirname(auditCsv), { recursive: true });
  const orderedRows = Array.from(auditRowsById.values()).sort((left, right) =>
    left.arxiv_id.localeCompare(right.arxiv_id),
  );
  const csvText =
    typeof csvStringify === "function"
      ? csvStringify(orderedRows, {
          header: true,
          columns: CSV_HEADERS,
        })
      : buildAuditCsvFallback(orderedRows);
  await fsp.writeFile(auditCsv, csvText, "utf8");
}

/**
 * Create the default runtime from locally installed dependencies.
 *
 * @returns {Promise<DownloadRuntime>}
 */
export async function createLocalRuntime() {
  const [puppeteerCoreModule, puppeteerExtraModule, stealthModule, csvSyncModule] =
    await Promise.all([
      import("puppeteer-core"),
      import("puppeteer-extra"),
      import("puppeteer-extra-plugin-stealth"),
      import("csv/sync"),
    ]);

  const puppeteerCore = puppeteerCoreModule.default ?? puppeteerCoreModule;
  const addExtra =
    puppeteerExtraModule.addExtra ??
    puppeteerExtraModule.default?.addExtra;
  if (typeof addExtra !== "function") {
    throw new Error("Could not resolve addExtra from puppeteer-extra.");
  }
  const StealthPlugin = stealthModule.default ?? stealthModule;
  const puppeteer = addExtra(puppeteerCore);
  puppeteer.use(StealthPlugin());

  return {
    puppeteer,
    csvParse: csvSyncModule.parse,
    csvStringify: csvSyncModule.stringify,
  };
}

/**
 * Resolve the temporary npm-exec node_modules root from PATH.
 *
 * @returns {string}
 */
export function findInjectedNodeModulesRoot() {
  const pathEntries = String(process.env.PATH ?? "").split(path.delimiter);
  for (const entry of pathEntries) {
    if (!entry.includes(`${path.sep}.npm${path.sep}_npx${path.sep}`)) {
      continue;
    }
    if (!entry.endsWith(`${path.sep}node_modules${path.sep}.bin`)) {
      continue;
    }
    return path.dirname(entry);
  }
  return "";
}

/**
 * Create a runtime from npm-exec injected packages.
 *
 * @returns {DownloadRuntime}
 */
export function createInjectedRuntime() {
  const nodeModulesRoot = findInjectedNodeModulesRoot();
  if (!nodeModulesRoot) {
    throw new Error("Could not find npm exec injected node_modules in PATH.");
  }

  const requireFromInjectedRoot = createRequire(
    path.join(nodeModulesRoot, "__bootstrap__.cjs"),
  );
  const puppeteerCore = requireFromInjectedRoot("puppeteer-core");
  const { addExtra } = requireFromInjectedRoot("puppeteer-extra");
  const StealthPlugin = requireFromInjectedRoot(
    "puppeteer-extra-plugin-stealth",
  );
  const { parse, stringify } = requireFromInjectedRoot("csv/sync");

  const puppeteer = addExtra(puppeteerCore);
  puppeteer.use(StealthPlugin());
  return {
    puppeteer,
    csvParse: parse,
    csvStringify: stringify,
  };
}

/**
 * Parse a content-disposition header filename.
 *
 * @param {string} contentDisposition
 * @param {string} arxivId
 * @returns {string}
 */
export function derivePdfFileName(contentDisposition, arxivId) {
  const match = /filename="([^"]+)"/i.exec(contentDisposition);
  if (match && match[1]) {
    return match[1];
  }
  return `${arxivId}.pdf`;
}

/**
 * Fetch a PDF within the browser context and return a base64 payload.
 *
 * @param {import("puppeteer-core").Page} page
 * @param {string} pdfUrl
 * @returns {Promise<{
 *   status: number,
 *   contentType: string,
 *   contentDisposition: string,
 *   bodyBase64: string,
 *   error: string,
 * }>}
 */
export async function fetchPdfPayloadInPage(page, pdfUrl) {
  return page.evaluate(async (url) => {
    try {
      const response = await fetch(url, {
        credentials: "include",
        redirect: "follow",
      });
      const contentType = response.headers.get("content-type") ?? "";
      const contentDisposition =
        response.headers.get("content-disposition") ?? "";
      const arrayBuffer = await response.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      let binary = "";
      const chunkSize = 0x8000;
      for (let offset = 0; offset < bytes.length; offset += chunkSize) {
        const chunk = bytes.subarray(offset, offset + chunkSize);
        binary += String.fromCharCode(...chunk);
      }

      return {
        status: response.status,
        contentType,
        contentDisposition,
        bodyBase64: btoa(binary),
        error: "",
      };
    } catch (error) {
      return {
        status: 0,
        contentType: "",
        contentDisposition: "",
        bodyBase64: "",
        error: error instanceof Error ? error.message : String(error ?? "Unknown error"),
      };
    }
  }, pdfUrl);
}

/**
 * Return whether a status should count as a browser-level failure.
 *
 * @param {string} status
 * @returns {boolean}
 */
export function isBrowserLevelFailure(status) {
  return (
    status === STATUS.ABS_TIMEOUT ||
    status === STATUS.PDF_TIMEOUT ||
    status === STATUS.PDF_RESPONSE_MISSING ||
    status === STATUS.NON_PDF_RESPONSE
  );
}

/**
 * Mark an existing PDF in the audit map.
 *
 * @param {Map<string, AuditRow>} auditRowsById
 * @param {DownloadRow} row
 * @param {string} pdfPath
 * @returns {Promise<void>}
 */
export async function markAlreadyPresent(auditRowsById, row, pdfPath) {
  const auditRow = createAuditRow(row);
  const stat = await fsp.stat(pdfPath);
  auditRow.status = STATUS.ALREADY_PRESENT;
  auditRow.pdf_found = true;
  auditRow.pdf_path = pdfPath;
  auditRow.downloaded_at = stat.mtime.toISOString();
  auditRowsById.set(row.arxivId, auditRow);
}

/**
 * Attempt to download one PDF through headless Chrome.
 *
 * @param {import("puppeteer-extra").PuppeteerExtra} puppeteer
 * @param {DownloadRow} row
 * @param {string} outputDir
 * @param {CliOptions} options
 * @returns {Promise<{auditRow: AuditRow, browserLevelFailure: boolean}>}
 */
export async function attemptDownload(puppeteer, row, outputDir, options) {
  const targetDirectory = buildArxivOutputDirectory(outputDir, row.arxivId);
  const auditRow = createAuditRow(row);
  await fsp.mkdir(targetDirectory, { recursive: true });

  let browser = null;
  let abstractPage = null;
  let pdfPage = null;
  try {
    browser = await puppeteer.launch({
      browser: "chrome",
      executablePath: options.chromeExecutable,
      headless: true,
      timeout: options.absTimeoutMs,
      args: [
        "--no-first-run",
        "--disable-dev-shm-usage",
        "--disable-features=Translate,OptimizationHints",
        "--lang=en-US,en",
      ],
    });

    abstractPage = await browser.newPage();
    abstractPage.setDefaultNavigationTimeout(options.absTimeoutMs);
    abstractPage.setDefaultTimeout(options.absTimeoutMs);
    const abstractResponse = await abstractPage.goto(auditRow.abs_url, {
      waitUntil: "domcontentloaded",
      timeout: options.absTimeoutMs,
    });
    auditRow.last_attempt_at = nowIso();
    if (!abstractResponse) {
      auditRow.status = STATUS.PDF_RESPONSE_MISSING;
      auditRow.last_error = "Abstract navigation did not return an HTTP response.";
      return { auditRow, browserLevelFailure: true };
    }

    auditRow.last_http_status = String(abstractResponse.status());
    if (abstractResponse.status() !== 200) {
      auditRow.status = STATUS.ABS_HTTP_ERROR;
      auditRow.last_error = `Abstract page returned HTTP ${abstractResponse.status()}.`;
      return { auditRow, browserLevelFailure: false };
    }

    const pdfHref = await abstractPage.evaluate(() => {
      const candidates = Array.from(document.querySelectorAll("a[href]"));
      for (const candidate of candidates) {
        const href = candidate.getAttribute("href") ?? "";
        if (href.includes("/pdf/")) {
          return href;
        }
      }
      return "";
    });
    if (!pdfHref) {
      auditRow.status = STATUS.PDF_RESPONSE_MISSING;
      auditRow.last_error = "Could not find a PDF link on the abstract page.";
      return { auditRow, browserLevelFailure: true };
    }

    auditRow.pdf_url = new URL(pdfHref, auditRow.abs_url).toString();

    pdfPage = await browser.newPage();
    pdfPage.setDefaultNavigationTimeout(options.pdfTimeoutMs);
    pdfPage.setDefaultTimeout(options.pdfTimeoutMs);
    const pdfContextResponse = await pdfPage.goto(auditRow.abs_url, {
      waitUntil: "domcontentloaded",
      timeout: options.pdfTimeoutMs,
    });
    auditRow.last_attempt_at = nowIso();
    if (!pdfContextResponse) {
      auditRow.status = STATUS.PDF_RESPONSE_MISSING;
      auditRow.last_error =
        "PDF context initialization did not return an HTTP response.";
      return { auditRow, browserLevelFailure: true };
    }
    const pdfFetchResult = await fetchPdfPayloadInPage(pdfPage, auditRow.pdf_url);
    if (pdfFetchResult.error) {
      auditRow.status = STATUS.PDF_RESPONSE_MISSING;
      auditRow.last_error = pdfFetchResult.error;
      return { auditRow, browserLevelFailure: true };
    }

    auditRow.last_http_status = String(pdfFetchResult.status);
    auditRow.last_content_type = pdfFetchResult.contentType;
    if (pdfFetchResult.status !== 200) {
      auditRow.status = STATUS.PDF_HTTP_ERROR;
      auditRow.last_error = `PDF request returned HTTP ${pdfFetchResult.status}.`;
      return { auditRow, browserLevelFailure: false };
    }
    if (!auditRow.last_content_type.toLowerCase().includes(PDF_CONTENT_TYPE)) {
      auditRow.status = STATUS.NON_PDF_RESPONSE;
      auditRow.last_error = `Unexpected content-type: ${auditRow.last_content_type || "missing"}.`;
      return { auditRow, browserLevelFailure: true };
    }

    const pdfBuffer = Buffer.from(pdfFetchResult.bodyBase64, "base64");
    if (!pdfBuffer || pdfBuffer.length === 0) {
      auditRow.status = STATUS.PDF_RESPONSE_MISSING;
      auditRow.last_error = "PDF response body was empty.";
      return { auditRow, browserLevelFailure: true };
    }
    if (!pdfBuffer.subarray(0, 4).equals(Buffer.from("%PDF"))) {
      auditRow.status = STATUS.NON_PDF_RESPONSE;
      auditRow.last_error = "Response body does not begin with a PDF signature.";
      return { auditRow, browserLevelFailure: true };
    }

    let finalPath = "";
    try {
      const fileName = derivePdfFileName(
        pdfFetchResult.contentDisposition,
        row.arxivId,
      );
      finalPath = path.join(targetDirectory, fileName);
      const temporaryPath = `${finalPath}.part`;
      await fsp.writeFile(temporaryPath, pdfBuffer);
      await fsp.rename(temporaryPath, finalPath);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : String(error ?? "Unknown error");
      auditRow.status = STATUS.WRITE_FAILED;
      auditRow.last_error = message;
      return { auditRow, browserLevelFailure: false };
    }

    auditRow.status = STATUS.DOWNLOADED;
    auditRow.pdf_found = true;
    auditRow.pdf_path = finalPath;
    auditRow.downloaded_at = nowIso();
    return { auditRow, browserLevelFailure: false };
  } catch (error) {
    const message =
      error instanceof Error ? error.message : String(error ?? "Unknown error");
    auditRow.last_attempt_at = nowIso();
    if (message.toLowerCase().includes("timeout")) {
      auditRow.status = auditRow.pdf_url ? STATUS.PDF_TIMEOUT : STATUS.ABS_TIMEOUT;
    } else if (message.toLowerCase().includes("write")) {
      auditRow.status = STATUS.WRITE_FAILED;
    } else {
      auditRow.status = auditRow.pdf_url
        ? STATUS.PDF_RESPONSE_MISSING
        : STATUS.ABS_HTTP_ERROR;
    }
    auditRow.last_error = message;
    return {
      auditRow,
      browserLevelFailure: isBrowserLevelFailure(auditRow.status),
    };
  } finally {
    await closePageQuietly(pdfPage);
    await closePageQuietly(abstractPage);
    await closeBrowserQuietly(browser);
  }
}

/**
 * Close a Puppeteer page without surfacing a cleanup error.
 *
 * @param {import("puppeteer-core").Page | null} page
 * @returns {Promise<void>}
 */
export async function closePageQuietly(page) {
  if (!page) {
    return;
  }
  try {
    await page.close({ runBeforeUnload: false });
  } catch (_error) {
    // Ignore cleanup failures.
  }
}

/**
 * Close a browser without surfacing a cleanup error.
 *
 * @param {import("puppeteer-core").Browser | null} browser
 * @returns {Promise<void>}
 */
export async function closeBrowserQuietly(browser) {
  if (!browser) {
    return;
  }
  try {
    await browser.close();
  } catch (_error) {
    // Ignore cleanup failures.
  }
}

/**
 * Run the downloader.
 *
 * @param {{argv?: string[], puppeteer: DownloadRuntime["puppeteer"], csvParse: DownloadRuntime["csvParse"], csvStringify: DownloadRuntime["csvStringify"]}} runtime
 * @returns {Promise<void>}
 */
export async function main(runtime) {
  const argv = runtime.argv ?? process.argv.slice(2);
  const options = parseCliOptions(argv);
  const downloadRows = await loadDownloadRows(
    options.inputCsv,
    runtime.csvParse,
    options.limit,
  );
  if (!downloadRows.length) {
    throw new Error("No arXiv IDs were found in the input CSV.");
  }

  await fsp.mkdir(options.outputDir, { recursive: true });
  const initialPdfCount = await countActualPdfFiles(options.outputDir);
  let potentialNewPdfCount = 0;
  for (const row of downloadRows) {
    const existingPdf = await findExistingPdf(
      buildArxivOutputDirectory(options.outputDir, row.arxivId),
    );
    if (!existingPdf) {
      potentialNewPdfCount += 1;
    }
  }
  const maxAchievableCount = initialPdfCount + potentialNewPdfCount;
  if (options.limit > 0 && options.targetCount > maxAchievableCount) {
    throw new Error(
      `Target count ${options.targetCount} cannot be reached with --limit ${options.limit}.`,
    );
  }

  /** @type {Map<string, AuditRow>} */
  const auditRowsById = new Map();
  for (const row of downloadRows) {
    const existingPdf = await findExistingPdf(
      buildArxivOutputDirectory(options.outputDir, row.arxivId),
    );
    if (existingPdf) {
      await markAlreadyPresent(auditRowsById, row, existingPdf);
    }
  }

  let passIndex = 0;
  while ((await countActualPdfFiles(options.outputDir)) < options.targetCount) {
    passIndex += 1;
    process.stdout.write(
      `[*] Starting pass ${passIndex}. Current PDF count: ${await countActualPdfFiles(options.outputDir)} / ${options.targetCount}\n`,
    );

    let processedSinceReset = 0;
    let consecutiveBrowserFailures = 0;
    for (const row of downloadRows) {
      const existingPdf = await findExistingPdf(
        buildArxivOutputDirectory(options.outputDir, row.arxivId),
      );
      if (existingPdf) {
        if (!auditRowsById.has(row.arxivId)) {
          await markAlreadyPresent(auditRowsById, row, existingPdf);
        }
        continue;
      }

      for (let attemptIndex = 1; attemptIndex <= DEFAULT_PASS_RETRY_COUNT; attemptIndex += 1) {
        const outcome = await attemptDownload(
          runtime.puppeteer,
          row,
          options.outputDir,
          options,
        );
        const auditRow =
          auditRowsById.get(row.arxivId) ?? createAuditRow(row);
        const attemptCount = auditRow.attempt_count + 1;
        outcome.auditRow.attempt_count = attemptCount;
        auditRowsById.set(row.arxivId, outcome.auditRow);
        await writeAuditCsv(auditRowsById, options.auditCsv, runtime.csvStringify);

        process.stdout.write(
          `    - ${row.arxivId} attempt ${attemptIndex}/${DEFAULT_PASS_RETRY_COUNT}: ${outcome.auditRow.status}\n`,
        );

        if (outcome.auditRow.pdf_found) {
          consecutiveBrowserFailures = 0;
          break;
        }

        if (outcome.browserLevelFailure) {
          consecutiveBrowserFailures += 1;
        } else {
          consecutiveBrowserFailures = 0;
        }

        if (consecutiveBrowserFailures >= DEFAULT_BROWSER_FAILURE_THRESHOLD) {
          process.stdout.write(
            `    - browser failure threshold reached after ${row.arxivId}; resetting state before continuing.\n`,
          );
          consecutiveBrowserFailures = 0;
          break;
        }
      }

      processedSinceReset += 1;
      if ((await countActualPdfFiles(options.outputDir)) >= options.targetCount) {
        break;
      }

      if (processedSinceReset >= DEFAULT_BROWSER_RESET_INTERVAL) {
        processedSinceReset = 0;
      }
      await sleep(randomJitterMilliseconds());
    }

    const pdfCount = await countActualPdfFiles(options.outputDir);
    process.stdout.write(
      `[*] Completed pass ${passIndex}. Current PDF count: ${pdfCount} / ${options.targetCount}\n`,
    );
    if (pdfCount >= options.targetCount) {
      break;
    }
    process.stdout.write(
      `[*] Sleeping ${DEFAULT_PASS_SLEEP_MS / 1000} seconds before retrying missing IDs.\n`,
    );
    await sleep(DEFAULT_PASS_SLEEP_MS);
  }

  await writeAuditCsv(auditRowsById, options.auditCsv, runtime.csvStringify);
  process.stdout.write(
    `[+] Target reached with ${await countActualPdfFiles(options.outputDir)} PDFs. Audit: ${options.auditCsv}\n`,
  );
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  printHelp();
  process.exit(0);
}

if (process.argv[1] && path.resolve(process.argv[1]) === SCRIPT_PATH) {
  try {
    let runtime = null;
    try {
      runtime = await createLocalRuntime();
    } catch (_localError) {
      runtime = createInjectedRuntime();
    }

    if (!runtime) {
      throw new Error("Could not initialize a Puppeteer runtime.");
    }
    await main(runtime);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : String(error ?? "Unknown error");
    process.stderr.write(`${message}\n`);
    process.stderr.write(
      `Run via npm exec if dependencies are not installed locally:\n${buildBootstrapCommand()} --input-csv ${DEFAULT_INPUT_CSV} --output-dir ${DEFAULT_OUTPUT_DIR} --audit-csv ${DEFAULT_AUDIT_CSV} --target-count ${DEFAULT_TARGET_COUNT}\n`,
    );
    process.exit(1);
  }
}
