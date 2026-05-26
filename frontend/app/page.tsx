"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Loader2,
  MessageSquare,
  PanelLeft,
  PanelLeftClose,
  Plus,
  Send,
  ShieldAlert,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  User,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  sources?: string[];
  needsHumanNote?: boolean;
  isStreaming?: boolean;
}

const SUGGESTED_QUESTIONS = [
  "What is the sick leave policy?",
  "How do I apply for maternity leave?",
  "What are the WFH guidelines?",
  "How to claim medical reimbursement?",
];

interface PendingApprovalAction {
  tool_name: string;
  action: string;
  reason: string;
  input?: { draft?: string; subject?: string; recipient?: string; message?: string };
}

interface PendingApproval {
  userMessage: string;
  actions: PendingApprovalAction[];
}

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1").replace(/\/chat$/, "");
const API_URL = `${API_BASE}/chat`;
const STREAM_URL = `${API_BASE}/chat/stream`;

function formatAssistantMessage(data: {
  answer?: string;
  sources?: Array<{ title?: string | null; source?: string | null }>;
}): ChatMessage {
  const assistantText = data.answer || "No response from assistant.";
  const sources = (data.sources || []).map((source) => {
    const title = source.title ? `${source.title}` : "Unknown title";
    const sourceId = source.source ? `${source.source}` : "unknown source";
    return `${title} — ${sourceId}`;
  });
  return {
    id: `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    role: "assistant",
    text: assistantText,
    sources,
  };
}

interface SessionInfo {
  session_id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "system",
      text: "Hi — I'm your HR policy assistant. Ask about leave, reimbursement, WFH, or say draft an email when you need help wording a request.",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [approvalAcknowledged, setApprovalAcknowledged] = useState(false);
  const [isConfirmingApproval, setIsConfirmingApproval] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Sidebar States
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [sessionsList, setSessionsList] = useState<SessionInfo[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Collapse sidebar initially on mobile screens
  useEffect(() => {
    if (typeof window !== "undefined" && window.innerWidth < 768) {
      setIsSidebarOpen(false);
    }
  }, []);

  const loadSessionsList = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/sessions`);
      if (!response.ok) return;
      const data = await response.json();
      setSessionsList(data.sessions || []);
    } catch {
      // Silently ignore to avoid disrupting chat workflow
    }
  }, []);

  useEffect(() => {
    void loadSessionsList();
  }, [loadSessionsList]);

  const startNewChat = () => {
    setSessionId(null);
    setMessages([
      {
        id: "welcome",
        role: "system",
        text: "Hi — I'm your HR policy assistant. Ask about leave, reimbursement, WFH, or say draft an email when you need help wording a request.",
      },
    ]);
    setInputValue("");
    setErrorMessage(null);
    setFeedback(null);
  };

  const switchSession = async (targetSessionId: string) => {
    setIsLoading(true);
    setErrorMessage(null);
    setFeedback(null);
    try {
      const response = await fetch(`${API_BASE}/sessions/${targetSessionId}`);
      if (!response.ok) {
        throw new Error("Unable to load session history.");
      }
      const data = await response.json();
      setSessionId(targetSessionId);

      const mapped = data.messages.map((m: any, index: number) => ({
        id: `${m.role}-${index}-${Date.now()}`,
        role: m.role,
        text: m.content,
      }));

      if (mapped.length === 0) {
        setMessages([
          {
            id: "welcome",
            role: "system",
            text: "Hi — I'm your HR policy assistant. Ask about leave, reimbursement, WFH, or say draft an email when you need help wording a request.",
          },
        ]);
      } else {
        setMessages(mapped);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to switch chat.");
    } finally {
      setIsLoading(false);
    }
  };

  const deleteSession = async (targetSessionId: string) => {
    try {
      const response = await fetch(`${API_BASE}/sessions/${targetSessionId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error("Failed to delete session.");
      }
      if (sessionId === targetSessionId) {
        startNewChat();
      }
      await loadSessionsList();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to delete chat.");
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const sendChatRequest = useCallback(
    async (messageText: string, approvedToolActions?: { tool_name: string; action: string }[]) => {
      const body: Record<string, unknown> = {
        message: messageText,
        access_level: "employee",
        session_id: sessionId,
      };
      if (approvedToolActions && approvedToolActions.length > 0) {
        body.approved_tool_actions = approvedToolActions;
      }

      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const responseBody = await response.json().catch(() => null);
        throw new Error(responseBody?.error?.message || "Unable to get response from backend.");
      }

      return response.json() as Promise<{
        answer?: string;
        sources?: Array<{ title?: string | null; source?: string | null }>;
        needs_human_confirmation?: boolean;
        approval_required_actions?: Array<{
          tool_name?: string;
          action?: string;
          reason?: string;
          input?: { draft?: string; subject?: string; recipient?: string; message?: string };
        }>;
        session_id?: string;
      }>;
    },
    [sessionId],
  );

  useEffect(() => {
    if (!pendingApproval) {
      return undefined;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPendingApproval(null);
        setApprovalAcknowledged(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pendingApproval]);

  const pushAssistantFromResponse = useCallback(
    (
      data: {
        answer?: string;
        sources?: Array<{ title?: string | null; source?: string | null }>;
        needs_human_confirmation?: boolean;
        approval_required_actions?: Array<{
          tool_name?: string;
          action?: string;
          reason?: string;
          input?: { draft?: string; subject?: string; recipient?: string; message?: string };
        }>;
        session_id?: string;
      },
      options?: { openApprovalModalFor?: string },
    ) => {
      if (data.session_id) setSessionId(data.session_id);
      const msg = formatAssistantMessage(data);
      const hasPendingActions = Array.isArray(data.approval_required_actions) && data.approval_required_actions.length > 0;
      msg.needsHumanNote = Boolean(data.needs_human_confirmation && !hasPendingActions);

      setMessages((current) => [...current, msg]);

      if (hasPendingActions && options?.openApprovalModalFor) {
        const raw = data.approval_required_actions!;
        setPendingApproval({
          userMessage: options.openApprovalModalFor,
          actions: raw.map((item) => ({
            tool_name: String(item.tool_name ?? "unknown"),
            action: String(item.action ?? "unknown"),
            reason: String(item.reason ?? "This action requires your confirmation."),
            input: item.input,
          })),
        });
        setApprovalAcknowledged(false);
      }
    },
    [setSessionId],
  );

  const sendStreamingMessage = async (text: string) => {
    const streamingId = `assistant-stream-${Date.now()}`;
    // NOTE: streaming message is NOT added here — it's added on the first token,
    // so "Thinking…" stays visible until real content starts arriving.

    try {
      const response = await fetch(STREAM_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          access_level: "employee",
          session_id: sessionId,
        }),
      });

      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || body?.error?.message || "Streaming request failed.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";
      let messageAdded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const line = part.startsWith("data: ") ? part.slice(6).trim() : part.trim();
          if (!line) continue;
          try {
            const event = JSON.parse(line) as {
              type: string;
              text?: string;
              sources?: Array<{ title?: string | null; source?: string | null }>;
              session_id?: string;
              needs_human_confirmation?: boolean;
              approval_required_actions?: Array<{
                tool_name?: string;
                action?: string;
                reason?: string;
                input?: { draft?: string; subject?: string; recipient?: string; message?: string };
              }>;
            };

            if (event.type === "token" && event.text) {
              fullText += event.text;
              if (!messageAdded) {
                // First token — replace "Thinking…" with the streaming message
                setMessages((prev) => [
                  ...prev,
                  { id: streamingId, role: "assistant", text: fullText, isStreaming: true },
                ]);
                messageAdded = true;
              } else {
                setMessages((prev) =>
                  prev.map((m) => (m.id === streamingId ? { ...m, text: fullText } : m)),
                );
              }
            } else if (event.type === "done") {
              if (event.session_id) setSessionId(event.session_id);
              const sources = (event.sources ?? []).map((s) => {
                const title = s.title ?? "Unknown title";
                const src = s.source ?? "unknown source";
                return `${title} — ${src}`;
              });
              const rawApprovals = event.approval_required_actions ?? [];
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === streamingId
                    ? {
                        ...m,
                        text: fullText || "No response.",
                        isStreaming: false,
                        sources,
                        needsHumanNote: (event.needs_human_confirmation ?? false) && rawApprovals.length === 0,
                      }
                    : m,
                ),
              );
              if (rawApprovals.length > 0) {
                setPendingApproval({
                  userMessage: text,
                  actions: rawApprovals.map((item) => ({
                    tool_name: String(item.tool_name ?? "unknown"),
                    action: String(item.action ?? "unknown"),
                    reason: String(item.reason ?? "This action requires your confirmation."),
                    input: item.input,
                  })),
                });
                setApprovalAcknowledged(false);
              }
            } else if (event.type === "error") {
              throw new Error((event as { message?: string }).message ?? "Stream error.");
            }
          } catch {
            // non-JSON line — skip
          }
        }
      }
    } catch (error) {
      // Remove the streaming message only if it was added
      setMessages((prev) => prev.filter((m) => m.id !== streamingId));
      setErrorMessage(error instanceof Error ? error.message : "Unexpected error occurred.");
    } finally {
      setIsLoading(false);
      void loadSessionsList();
    }
  };

  const sendMessage = async (overrideText?: string) => {
    const text = overrideText ?? inputValue.trim();
    if (!text) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      text,
    };

    setMessages((current) => [...current, userMessage]);
    setInputValue("");
    setErrorMessage(null);
    setFeedback(null);
    setIsLoading(true);

    await sendStreamingMessage(text);
  };

  const confirmPendingApprovals = async () => {
    if (!pendingApproval || !approvalAcknowledged) {
      return;
    }

    setIsConfirmingApproval(true);
    setErrorMessage(null);

    try {
      const approvedPayload = pendingApproval.actions.map((action) => ({
        tool_name: action.tool_name,
        action: action.action,
      }));
      const data = await sendChatRequest(pendingApproval.userMessage, approvedPayload);
      setPendingApproval(null);
      setApprovalAcknowledged(false);
      pushAssistantFromResponse(data, { openApprovalModalFor: pendingApproval.userMessage });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Unexpected error occurred.");
    } finally {
      setIsConfirmingApproval(false);
      void loadSessionsList();
    }
  };

  const submitFeedback = (value: "helpful" | "not_helpful") => {
    setFeedback(value === "helpful" ? "Thanks — glad it helped." : "Thanks — we'll use this to improve.");
  };

  const friendlyToolLabel = (toolName: string, action: string) => {
    if (toolName === "email_draft" && action === "send") {
      return "Send email";
    }
    if (toolName === "hr_ticket" && action === "create") {
      return "Create HR ticket";
    }
    return `${toolName} · ${action}`;
  };

  const onComposerKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  return (
    <main className="flex h-screen w-full bg-[#0c1222] text-slate-100 overflow-hidden">
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.35]"
        aria-hidden
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(45,212,191,0.12), transparent), radial-gradient(ellipse 60% 40% at 100% 50%, rgba(56,189,248,0.06), transparent)",
        }}
      />

      {pendingApproval ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#0c1222]/85 p-4 backdrop-blur-md"
          role="dialog"
          aria-modal="true"
          aria-labelledby="approval-modal-title"
        >
          <div className="relative w-full max-w-md rounded-2xl border border-white/10 bg-slate-900/95 p-6 shadow-2xl shadow-black/40 ring-1 ring-teal-500/20">
            <button
              type="button"
              onClick={() => {
                setPendingApproval(null);
                setApprovalAcknowledged(false);
              }}
              className="absolute right-3 top-3 rounded-full p-2 text-slate-400 transition hover:bg-white/10 hover:text-white"
              aria-label="Close"
            >
              <X className="h-5 w-5" />
            </button>
            <div className="flex items-start gap-3 pr-8">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-500/15 text-amber-400">
                <ShieldAlert className="h-5 w-5" aria-hidden />
              </div>
              <div>
                <h2 id="approval-modal-title" className="text-lg font-semibold tracking-tight text-white">
                  Confirm sensitive action
                </h2>
                <p className="mt-1 text-sm leading-relaxed text-slate-400">
                  Review what will run, then confirm. We&apos;ll resend your last message with approval.
                </p>
              </div>
            </div>

            <ul className="mt-5 space-y-3">
              {pendingApproval.actions.map((item) => (
                <li
                  key={`${item.tool_name}-${item.action}`}
                  className="rounded-xl border border-amber-500/25 bg-amber-500/[0.07] px-4 py-3"
                >
                  <p className="text-sm font-medium text-amber-100/95">{friendlyToolLabel(item.tool_name, item.action)}</p>
                  <p className="mt-1 text-sm text-slate-300">{item.reason}</p>
                  {item.input?.draft && (
                    <div className="mt-3 rounded-lg border border-white/10 bg-slate-950/70 p-3">
                      {item.input.recipient && (
                        <p className="mb-0.5 text-xs text-slate-400">
                          <span className="font-medium text-slate-300">To:</span> {item.input.recipient}
                        </p>
                      )}
                      {item.input.subject && (
                        <p className="mb-2 text-xs text-slate-400">
                          <span className="font-medium text-slate-300">Subject:</span> {item.input.subject}
                        </p>
                      )}
                      <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-300">
                        {item.input.draft.replace(/^To:.*\n\n/, "").replace(/^Subject:.*\n\n/, "")}
                      </pre>
                    </div>
                  )}
                </li>
              ))}
            </ul>

            <label className="mt-5 flex cursor-pointer items-start gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4 text-sm leading-snug text-slate-200">
              <input
                type="checkbox"
                checked={approvalAcknowledged}
                onChange={(event) => setApprovalAcknowledged(event.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-slate-500 bg-slate-900 text-teal-500 focus:ring-teal-500/40"
              />
              <span>I have reviewed the content above and approve sending it.</span>
            </label>

            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setPendingApproval(null);
                  setApprovalAcknowledged(false);
                }}
                className="rounded-xl border border-white/10 px-4 py-2.5 text-sm font-medium text-slate-300 transition hover:bg-white/5"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void confirmPendingApprovals()}
                disabled={!approvalAcknowledged || isConfirmingApproval}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-teal-900/30 transition hover:from-teal-400 hover:to-cyan-400 disabled:cursor-not-allowed disabled:opacity-45"
              >
                {isConfirmingApproval ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Confirm &amp; run
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {/* COLLAPSIBLE SIDEBAR */}
      <aside
        className={`relative z-20 flex h-full flex-col border-r border-white/5 bg-[#080d1a]/85 backdrop-blur-md transition-all duration-300 ${
          isSidebarOpen ? "w-64 sm:w-72" : "w-0 overflow-hidden border-r-0"
        }`}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between border-b border-white/5 p-4">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-teal-400 to-cyan-600 shadow-md">
              <Sparkles className="h-4 w-4 text-white" />
            </div>
            <span className="font-semibold text-white tracking-tight">HR Assistant</span>
          </div>
          <button
            type="button"
            onClick={() => setIsSidebarOpen(false)}
            className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-white/5 transition"
            title="Collapse sidebar"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <button
            type="button"
            onClick={startNewChat}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm font-semibold text-slate-200 transition hover:border-teal-500/40 hover:bg-teal-500/10 hover:text-white"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-thin select-none">
          <p className="px-3 mb-2 text-xs uppercase tracking-wider text-slate-500 font-semibold">Recent Chats</p>
          {sessionsList.length === 0 ? (
            <p className="px-3 py-4 text-center text-xs text-slate-600 italic">No recent chats</p>
          ) : (
            sessionsList.map((session) => (
              <div
                key={session.session_id}
                className={`group relative flex items-center justify-between rounded-xl px-3 py-2.5 text-sm transition cursor-pointer ${
                  sessionId === session.session_id
                    ? "bg-teal-500/10 text-teal-200 border border-teal-500/20"
                    : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200 border border-transparent"
                }`}
                onClick={() => void switchSession(session.session_id)}
              >
                <div className="flex min-w-0 flex-1 items-center gap-2">
                  <MessageSquare className="h-4 w-4 shrink-0 text-slate-500 group-hover:text-teal-400" />
                  <span className="truncate pr-5 font-medium">{session.title}</span>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    void deleteSession(session.session_id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-rose-400 rounded transition absolute right-2 bg-slate-900/90 hover:bg-slate-800"
                  title="Delete chat"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="border-t border-white/5 p-4 flex items-center justify-between bg-slate-950/20">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-slate-800 flex items-center justify-center text-xs text-teal-400 font-semibold ring-1 ring-white/10">
              HR
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-200">Employee Portal</p>
              <p className="text-[10px] text-slate-500">Access: Level 1</p>
            </div>
          </div>
          <Link
            href="/admin/documents"
            className="rounded-lg border border-white/10 bg-white/[0.04] p-1.5 text-slate-400 transition hover:border-teal-500/40 hover:bg-teal-500/10 hover:text-white"
            title="Admin Panel"
          >
            <User className="h-4 w-4" />
          </Link>
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <div className="flex flex-1 flex-col h-full overflow-hidden relative">
        <header className="relative z-10 border-b border-white/5 bg-[#0c1222]/80 backdrop-blur-lg">
          <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
            <div className="flex min-w-0 items-center gap-3">
              {!isSidebarOpen && (
                <button
                  type="button"
                  onClick={() => setIsSidebarOpen(true)}
                  className="p-2 text-slate-400 hover:text-slate-200 rounded-xl hover:bg-white/5 transition shrink-0"
                  title="Show sidebar"
                >
                  <PanelLeft className="h-5 w-5" />
                </button>
              )}
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 shadow-md shadow-teal-900/40">
                <Sparkles className="h-5 w-5 text-white" aria-hidden />
               </div>
               <div className="min-w-0">
                 <h1 className="truncate text-base font-semibold tracking-tight text-white sm:text-lg">HR Policy Assistant</h1>
                 <p className="truncate text-xs text-slate-400 sm:text-sm">Policy Q&amp;A · drafts · ticket help</p>
               </div>
             </div>
             <Link
               href="/admin/documents"
               className="shrink-0 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-teal-500/40 hover:bg-teal-500/10 hover:text-white sm:px-4 sm:text-sm"
             >
               Admin
             </Link>
           </div>
         </header>

         <div className="relative z-10 flex-1 overflow-y-auto px-4 pb-4 pt-6 sm:px-6">
           <div className="mx-auto max-w-3xl space-y-5">
             {messages.map((message) => {
               if (message.role === "system") {
                 return (
                   <div key={message.id} className="flex flex-col items-center gap-3 px-2">
                     <p className="max-w-lg text-center text-sm leading-relaxed text-slate-500">{message.text}</p>
                     {message.id === "welcome" && (
                       <div className="flex flex-wrap justify-center gap-2">
                         {SUGGESTED_QUESTIONS.map((q) => (
                           <button
                             key={q}
                             type="button"
                             disabled={isLoading}
                             onClick={() => void sendMessage(q)}
                             className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-slate-400 transition hover:border-teal-500/30 hover:bg-teal-500/10 hover:text-teal-200 disabled:pointer-events-none disabled:opacity-40"
                           >
                             {q}
                           </button>
                         ))}
                       </div>
                     )}
                   </div>
                 );
               }

               if (message.role === "user") {
                 return (
                   <div key={message.id} className="flex justify-end gap-2">
                     <div className="max-w-[min(85%,28rem)] rounded-2xl rounded-br-md bg-gradient-to-br from-teal-500 to-cyan-600 px-4 py-3 text-[15px] leading-relaxed text-white shadow-lg shadow-teal-950/40">
                       <p className="whitespace-pre-wrap">{message.text}</p>
                     </div>
                     <div
                       className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-700 text-slate-200 sm:flex"
                       aria-hidden
                     >
                       <User className="h-4 w-4" />
                     </div>
                   </div>
                 );
               }

               return (
                 <div key={message.id} className="flex justify-start gap-2">
                   <div
                     className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-800 text-teal-400 ring-1 ring-white/10 sm:flex"
                     aria-hidden
                   >
                     <Sparkles className="h-4 w-4" />
                   </div>
                   <div className="max-w-[min(90%,32rem)] space-y-3">
                     <div className="rounded-2xl rounded-bl-md border border-white/10 bg-slate-800/70 px-4 py-3 text-[15px] leading-relaxed text-slate-200 shadow-lg shadow-black/20 backdrop-blur-sm">
                       <div className="prose prose-invert prose-sm max-w-none
                         prose-p:my-1 prose-p:leading-relaxed
                         prose-ul:my-1 prose-ul:pl-4 prose-ol:my-1 prose-ol:pl-4
                         prose-li:my-0.5
                         prose-strong:text-white prose-strong:font-semibold
                         prose-code:rounded prose-code:bg-slate-900 prose-code:px-1.5 prose-code:py-0.5 prose-code:text-xs prose-code:text-teal-300 prose-code:before:content-none prose-code:after:content-none
                         prose-table:text-sm prose-th:text-teal-300 prose-th:font-semibold prose-td:text-slate-300
                         prose-headings:text-white prose-headings:font-semibold
                       ">
                         <ReactMarkdown remarkPlugins={[remarkGfm]}>
                           {message.text || (message.isStreaming ? "" : "No response.")}
                         </ReactMarkdown>
                         {message.isStreaming && (
                           <span className="ml-0.5 inline-block h-[0.9em] w-0.5 animate-pulse rounded-sm bg-teal-400 align-middle" />
                         )}
                       </div>
                       {message.needsHumanNote ? (
                         <div className="mt-3 flex gap-2 rounded-lg border border-amber-500/20 bg-amber-500/[0.08] px-3 py-2 text-xs text-amber-100/90">
                           <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" aria-hidden />
                           <span>Please verify with HR for your specific situation — this answer may not cover every case.</span>
                         </div>
                       ) : null}
                     </div>
                     {message.sources && message.sources.length > 0 ? (
                       <div className="flex flex-wrap gap-2 pl-0.5 sm:pl-0">
                         {message.sources.map((source, index) => (
                           <span
                             key={`${message.id}-src-${index}`}
                             className="inline-flex max-w-full items-center gap-1 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-slate-400"
                             title={source}
                           >
                             <BookOpen className="h-3 w-3 shrink-0 text-teal-500/80" aria-hidden />
                             <span className="truncate">{source}</span>
                           </span>
                         ))}
                       </div>
                     ) : null}
                   </div>
                 </div>
               );
             })}

             {isLoading && !messages.some((m) => m.isStreaming && m.text.length > 0) ? (
               <div className="flex justify-start gap-2 pl-0 sm:pl-11">
                 <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-slate-800/50 px-4 py-3 text-sm text-slate-400">
                   <Loader2 className="h-4 w-4 animate-spin text-teal-400" />
                   Thinking…
                 </div>
               </div>
             ) : null}

             <div ref={messagesEndRef} />
           </div>
         </div>

         <div className="relative z-10 border-t border-white/5 bg-[#0c1222]/90 backdrop-blur-xl">
           <div className="mx-auto max-w-3xl space-y-3 px-4 py-4 sm:px-6">
             {errorMessage ? (
               <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">{errorMessage}</div>
             ) : null}

             <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
               <label className="sr-only" htmlFor="chat-input">
                 Ask a question
               </label>
               <textarea
                 id="chat-input"
                 rows={1}
                 className="max-h-40 min-h-[52px] flex-1 resize-y rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-[15px] text-slate-100 placeholder:text-slate-500 outline-none ring-teal-500/0 transition focus:border-teal-500/50 focus:ring-2 focus:ring-teal-500/20"
                 placeholder="Ask in any language… e.g. sick leave process, reimbursement steps"
                 value={inputValue}
                 onChange={(event) => setInputValue(event.target.value)}
                 onKeyDown={onComposerKeyDown}
                 disabled={isLoading || !!pendingApproval}
               />
               <button
                 type="button"
                 onClick={() => void sendMessage()}
                 disabled={isLoading || !!pendingApproval || !inputValue.trim()}
                 className="inline-flex h-[52px] shrink-0 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-teal-500 to-cyan-500 px-6 text-sm font-semibold text-slate-950 shadow-lg shadow-teal-900/25 transition hover:from-teal-400 hover:to-cyan-400 disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto"
               >
                 {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
                 Send
               </button>
             </div>
             <p className="text-center text-[11px] text-slate-600 sm:text-left">Enter to send · Shift+Enter for new line</p>

             <div className="flex flex-col items-center justify-between gap-3 border-t border-white/5 pt-3 sm:flex-row">
               <span className="text-xs text-slate-500">Was the last answer useful?</span>
               <div className="flex gap-2">
                 <button
                   type="button"
                   onClick={() => submitFeedback("helpful")}
                   className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-teal-500/30 hover:bg-teal-500/10 hover:text-teal-100"
                 >
                   <ThumbsUp className="h-3.5 w-3.5" /> Yes
                 </button>
                 <button
                   type="button"
                   onClick={() => submitFeedback("not_helpful")}
                   className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-slate-500/40 hover:bg-white/[0.08]"
                 >
                   <ThumbsDown className="h-3.5 w-3.5" /> No
                 </button>
               </div>
             </div>

             {feedback ? <p className="text-center text-xs text-teal-400/90 sm:text-left">{feedback}</p> : null}
           </div>
         </div>
       </div>
     </main>
   );
 }
