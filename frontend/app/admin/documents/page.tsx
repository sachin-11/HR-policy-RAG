"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from "react";
import { ArrowLeft, CheckCircle2, Eye, EyeOff, FileUp, KeyRound, Loader2, RefreshCw, Shield } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
const TOKEN_KEY = "hr_policy_admin_token";

interface DocumentRow {
  name: string;
  path: string;
  size_bytes: number;
  modified_at: string;
}

interface IndexingStatus {
  job_state: string;
  started_at: string | null;
  finished_at: string | null;
  message: string;
  error_detail: string | null;
  document_count: number;
  chunk_count: number;
  vector_count: number | null;
  vector_store_provider: string;
  embedding_provider: string;
}

export default function AdminDocumentsPage() {
  const [token, setToken] = useState("");
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [rawDocsDir, setRawDocsDir] = useState<string>("");
  const [indexStatus, setIndexStatus] = useState<IndexingStatus | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [reindexError, setReindexError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isReindexing, setIsReindexing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [autoReindex, setAutoReindex] = useState(true);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [adminPassword, setAdminPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [regenError, setRegenError] = useState<string | null>(null);

  const tokenExpired =
    listError?.toLowerCase().includes("expired") ||
    listError?.toLowerCase().includes("invalid authentication") ||
    false;

  useEffect(() => {
    const existing = typeof window !== "undefined" ? window.localStorage.getItem(TOKEN_KEY) : null;
    if (existing) {
      setToken(existing);
    }
  }, []);

  const authHeaders = useMemo(() => {
    if (!token.trim()) {
      return null;
    }
    return {
      Authorization: `Bearer ${token.trim()}`,
    };
  }, [token]);

  const persistToken = (value: string) => {
    setToken(value);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(TOKEN_KEY, value);
    }
  };

  const loadData = useCallback(async () => {
    if (!authHeaders) {
      setDocuments([]);
      setRawDocsDir("");
      setIndexStatus(null);
      return;
    }

    setIsLoading(true);
    setListError(null);

    try {
      const [docsRes, statusRes] = await Promise.all([
        fetch(`${API_BASE}/documents`, { headers: authHeaders }),
        fetch(`${API_BASE}/documents/indexing-status`, { headers: authHeaders }),
      ]);

      if (!docsRes.ok) {
        const body = await docsRes.json().catch(() => null);
        throw new Error(body?.error?.message || body?.detail || "Unable to load documents.");
      }
      if (!statusRes.ok) {
        const body = await statusRes.json().catch(() => null);
        throw new Error(body?.error?.message || body?.detail || "Unable to load indexing status.");
      }

      const docsBody = await docsRes.json();
      const statusBody = await statusRes.json();

      setDocuments(docsBody.documents || []);
      setRawDocsDir(docsBody.raw_docs_dir || "");
      setIndexStatus(statusBody);
    } catch (error) {
      setListError(error instanceof Error ? error.message : "Unexpected error while loading data.");
    } finally {
      setIsLoading(false);
    }
  }, [authHeaders]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  useEffect(() => {
    if (!authHeaders || !indexStatus || indexStatus.job_state !== "running") {
      return undefined;
    }

    const handle = window.setInterval(() => {
      void loadData();
    }, 2000);

    return () => window.clearInterval(handle);
  }, [authHeaders, indexStatus?.job_state, loadData]);

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !authHeaders) {
      return;
    }

    setUploadError(null);
    setUploadSuccess(null);
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/documents/upload`, {
        method: "POST",
        headers: authHeaders,
        body: formData,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.error?.message || body?.detail || "Upload failed.");
      }

      await loadData();

      if (autoReindex) {
        setUploadSuccess(`"${file.name}" uploaded — starting re-index…`);
        await handleReindex();
      } else {
        setUploadSuccess(`"${file.name}" uploaded successfully. Run re-index when ready.`);
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed unexpectedly.");
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  };

  const handleReindex = async () => {
    if (!authHeaders) {
      return;
    }

    setReindexError(null);
    setIsReindexing(true);

    try {
      const response = await fetch(`${API_BASE}/documents/reindex`, {
        method: "POST",
        headers: authHeaders,
      });

      if (response.status === 409) {
        setReindexError("Indexing is already running.");
        return;
      }

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.error?.message || body?.detail || "Re-index request failed.");
      }

      await loadData();
    } catch (error) {
      setReindexError(error instanceof Error ? error.message : "Unexpected error during re-index.");
    } finally {
      setIsReindexing(false);
    }
  };

  const handleRegenerateToken = async () => {
    if (!adminPassword.trim()) return;
    setIsRegenerating(true);
    setRegenError(null);
    setListError(null);
    try {
      const response = await fetch(`${API_BASE}/auth/admin-token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: adminPassword.trim() }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || body?.error?.message || "Could not generate token.");
      }
      const data = (await response.json()) as { token: string };
      persistToken(data.token);
    } catch (err) {
      setRegenError(err instanceof Error ? err.message : "Unexpected error.");
    } finally {
      setIsRegenerating(false);
    }
  };

  const statusBadge =
    indexStatus?.job_state === "running"
      ? "bg-amber-500/15 text-amber-200 border-amber-500/40"
      : indexStatus?.job_state === "success"
      ? "bg-emerald-500/15 text-emerald-200 border-emerald-500/40"
      : indexStatus?.job_state === "failed"
      ? "bg-rose-500/15 text-rose-200 border-rose-500/40"
      : "bg-slate-800 text-slate-200 border-slate-700";

  const canChooseFile = Boolean(authHeaders) && !isUploading;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-8">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.25em] text-slate-500">Module 12</p>
            <h1 className="text-3xl font-semibold">Admin · Documents</h1>
            <p className="text-slate-400">Upload policy files, monitor indexing, and re-build the vector index.</p>
          </div>
          <Link
            href="/"
            className="inline-flex items-center gap-2 rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200 transition hover:border-cyan-500/60 hover:text-white"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to chat
          </Link>
        </div>

        <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl shadow-slate-950/30">
          <div className="mb-4 flex items-start gap-3">
            <Shield className="mt-1 h-5 w-5 text-cyan-400" />
            <div>
              <h2 className="text-lg font-semibold">Admin authentication</h2>
              <p className="text-sm text-slate-400">
                Paste your Bearer token, or enter the admin password below to generate one automatically.
              </p>
            </div>
          </div>

          {/* Expired token warning */}
          {tokenExpired && (
            <div className="mb-4 flex items-center gap-3 rounded-2xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
              <KeyRound className="h-4 w-4 shrink-0 text-amber-400" />
              <span className="flex-1">Token expired — enter admin password below and click <strong>Regenerate</strong>.</span>
            </div>
          )}

          {/* Current JWT token input */}
          <label className="sr-only" htmlFor="admin-token">Admin token</label>
          <div className="relative">
            <input
              id="admin-token"
              type={showPassword ? "text" : "password"}
              autoComplete="off"
              value={token}
              onChange={(event) => persistToken(event.target.value)}
              className="w-full rounded-2xl border border-slate-800 bg-slate-950 px-4 py-3 pr-11 text-sm text-slate-100 outline-none transition focus:border-cyan-500"
              placeholder="Bearer token (admin role)"
            />
            <button
              type="button"
              onClick={() => setShowPassword((v) => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
              tabIndex={-1}
              aria-label={showPassword ? "Hide token" : "Show token"}
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>

          {/* Admin password + regenerate */}
          <div className="mt-3 flex gap-2">
            <input
              type="password"
              autoComplete="off"
              value={adminPassword}
              onChange={(e) => setAdminPassword(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void handleRegenerateToken(); }}
              placeholder="Admin password (to regenerate token)"
              className="flex-1 rounded-2xl border border-slate-800 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 outline-none transition focus:border-cyan-500"
            />
            <button
              type="button"
              onClick={() => void handleRegenerateToken()}
              disabled={!adminPassword.trim() || isRegenerating}
              className="inline-flex items-center gap-2 rounded-full bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isRegenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <KeyRound className="h-4 w-4" />}
              Regenerate
            </button>
          </div>

          {regenError && (
            <div className="mt-2 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-2.5 text-sm text-rose-100">
              {regenError}
            </div>
          )}

          <div className="mt-3 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void loadData()}
              disabled={!authHeaders || isLoading}
              className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-100 transition hover:border-cyan-500/60 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              Refresh data
            </button>
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">Indexing status</h2>
                <p className="text-sm text-slate-500">Latest job plus live vector counts (local store).</p>
              </div>
              <span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase ${statusBadge}`}>
                {indexStatus?.job_state ?? "idle"}
              </span>
            </div>

            {listError ? (
              <div className="rounded-2xl border border-rose-500/50 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                {listError}
              </div>
            ) : null}

            <dl className="mt-4 space-y-3 text-sm">
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Message</dt>
                <dd className="text-right text-slate-200">{indexStatus?.message || "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Vectors</dt>
                <dd className="text-right text-slate-200">{indexStatus?.vector_count ?? "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Chunks (last job)</dt>
                <dd className="text-right text-slate-200">{indexStatus?.chunk_count ?? "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Documents (last job)</dt>
                <dd className="text-right text-slate-200">{indexStatus?.document_count ?? "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Vector store</dt>
                <dd className="text-right text-slate-200">{indexStatus?.vector_store_provider || "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Embeddings</dt>
                <dd className="text-right text-slate-200">{indexStatus?.embedding_provider || "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Started</dt>
                <dd className="text-right text-slate-200">{indexStatus?.started_at || "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500">Finished</dt>
                <dd className="text-right text-slate-200">{indexStatus?.finished_at || "—"}</dd>
              </div>
            </dl>

            {indexStatus?.error_detail ? (
              <div className="mt-4 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                {indexStatus.error_detail}
              </div>
            ) : null}

            <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-medium text-slate-100">Re-index all documents</p>
                <p className="text-xs text-slate-500">Chunks, embeds, and upserts into the configured vector store.</p>
              </div>
              <button
                type="button"
                onClick={() => void handleReindex()}
                disabled={!authHeaders || isReindexing || indexStatus?.job_state === "running"}
                className="inline-flex items-center justify-center gap-2 rounded-full bg-cyan-500 px-5 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isReindexing || indexStatus?.job_state === "running" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
                Re-index
              </button>
            </div>
            {reindexError ? (
              <div className="mt-3 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                {reindexError}
              </div>
            ) : null}
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">Upload document</h2>
                <p className="text-sm text-slate-500">Supported: .md, .txt, .pdf (max 10 MB).</p>
              </div>
              <label className="flex cursor-pointer items-center gap-2 rounded-full border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-300 transition hover:border-cyan-500/50 select-none">
                <input
                  type="checkbox"
                  checked={autoReindex}
                  onChange={(e) => setAutoReindex(e.target.checked)}
                  className="accent-cyan-400"
                />
                Auto re-index
              </label>
            </div>
            <label
              htmlFor={canChooseFile ? "admin-file-upload" : undefined}
              className={`mt-6 flex flex-col items-center justify-center gap-3 rounded-3xl border border-dashed bg-slate-950/70 px-4 py-10 text-center transition ${
                canChooseFile
                  ? "cursor-pointer border-slate-700 hover:border-cyan-500/50"
                  : "cursor-pointer border-slate-700/80 opacity-60 hover:border-slate-600"
              }`}
              onClick={() => {
                if (!canChooseFile) {
                  document.getElementById("admin-token")?.focus();
                }
              }}
              title={canChooseFile ? undefined : "Paste an admin Bearer token above to enable uploads"}
            >
              <FileUp className={`h-6 w-6 ${canChooseFile ? "text-cyan-300" : "text-slate-500"}`} />
              <div>
                <p className="text-sm font-medium text-slate-100">Choose a file</p>
                <p className="text-xs text-slate-500">
                  {autoReindex
                    ? "Upload karega aur automatically re-index shuru karega."
                    : "Saves to backend raw docs folder · run re-index after upload."}
                </p>
                {!canChooseFile && !isUploading ? (
                  <p className="mt-2 text-xs font-medium text-amber-200/90">
                    Paste an admin token in the box above first — the file picker is disabled until then.
                  </p>
                ) : null}
              </div>
              <input
                id="admin-file-upload"
                type="file"
                className="sr-only"
                accept=".md,.markdown,.txt,.pdf"
                onChange={(event) => void handleUpload(event)}
                disabled={!canChooseFile}
                tabIndex={-1}
              />
            </label>

            {uploadSuccess && !uploadError ? (
              <div className="mt-4 flex items-start gap-2 rounded-2xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                {uploadSuccess}
              </div>
            ) : null}

            {uploadError ? (
              <div className="mt-4 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                {uploadError}
              </div>
            ) : null}

            {isUploading ? (
              <p className="mt-3 text-xs text-slate-400">
                <Loader2 className="mr-2 inline h-3 w-3 animate-spin" />
                Uploading…
              </p>
            ) : null}
          </div>
        </section>

        <section className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
          <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold">Documents in raw folder</h2>
              <p className="text-sm text-slate-500">{rawDocsDir || "Set an admin token to load the directory path."}</p>
            </div>
            <span className="text-xs uppercase tracking-[0.2em] text-slate-600">{documents.length} files</span>
          </div>

          <div className="overflow-hidden rounded-2xl border border-slate-800">
            <table className="min-w-full divide-y divide-slate-800 text-sm">
              <thead className="bg-slate-950/80 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Path</th>
                  <th className="px-4 py-3">Size</th>
                  <th className="px-4 py-3">Modified</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-900">
                {documents.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-4 py-6 text-center text-slate-500">
                      {authHeaders ? "No supported documents found yet." : "Add an admin token to view files."}
                    </td>
                  </tr>
                ) : (
                  documents.map((doc) => (
                    <tr key={doc.path} className="bg-slate-950/40">
                      <td className="px-4 py-3 font-medium text-slate-100">{doc.name}</td>
                      <td className="px-4 py-3 text-slate-400">{doc.path}</td>
                      <td className="px-4 py-3 text-slate-300">{(doc.size_bytes / 1024).toFixed(1)} KB</td>
                      <td className="px-4 py-3 text-slate-400">{new Date(doc.modified_at).toLocaleString()}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}
