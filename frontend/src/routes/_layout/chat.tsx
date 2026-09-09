import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  Bot,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  Loader2,
  MessageSquare,
  Plus,
  Send,
  Sparkles,
  Square,
  Trash2,
  User as UserIcon,
} from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import {
  type ChatMessagePublic,
  ChatService,
  type ChatSessionPublic,
  type ChatSource,
} from "@/services/ai"

export const Route = createFileRoute("/_layout/chat")({
  component: ChatPage,
  head: () => ({
    meta: [{ title: "AI Chat & Streaming RAG - FastAPI Template" }],
  }),
})

function ChatPage() {
  const queryClient = useQueryClient()
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [inputText, setInputText] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState("")
  const [streamingSources, setStreamingSources] = useState<ChatSource[]>([])
  const [expandedSources, setExpandedSources] = useState<
    Record<string, boolean>
  >({})

  const abortControllerRef = useRef<AbortController | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Fetch all chat sessions
  const { data: sessionsData, isLoading: isLoadingSessions } = useQuery({
    queryKey: ["chatSessions"],
    queryFn: () => ChatService.getSessions(0, 50),
  })

  const sessions = sessionsData?.data || []

  // Auto-select first session if none selected
  useEffect(() => {
    if (!activeSessionId && sessions.length > 0) {
      setActiveSessionId(sessions[0].id)
    }
  }, [sessions, activeSessionId])

  // Fetch active session messages
  const { data: activeSession, isLoading: isLoadingMessages } = useQuery({
    queryKey: ["chatSession", activeSessionId],
    queryFn: () =>
      activeSessionId ? ChatService.getSession(activeSessionId) : null,
    enabled: Boolean(activeSessionId),
  })

  // Auto-scroll on new messages or streaming tokens
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [])

  // Create Session Mutation
  const createSessionMutation = useMutation({
    mutationFn: (title?: string) => ChatService.createSession(title),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: ["chatSessions"] })
      setActiveSessionId(newSession.id)
      toast.success("New chat conversation started")
    },
    onError: (error: Error) => {
      toast.error(error.message || "Failed to create chat session")
    },
  })

  // Delete Session Mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (id: string) => ChatService.deleteSession(id),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: ["chatSessions"] })
      if (activeSessionId === deletedId) {
        const remaining = sessions.filter((s) => s.id !== deletedId)
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null)
      }
      toast.success("Chat session deleted")
    },
    onError: (error: Error) => {
      toast.error(error.message || "Failed to delete chat session")
    },
  })

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setIsStreaming(false)
      toast.info("Generation halted by user (disconnect detected on server)")
      queryClient.invalidateQueries({
        queryKey: ["chatSession", activeSessionId],
      })
      queryClient.invalidateQueries({ queryKey: ["aiUsage"] })
    }
  }

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!inputText.trim() || isStreaming) return

    let currentSessionId = activeSessionId
    if (!currentSessionId) {
      try {
        const newSession = await createSessionMutation.mutateAsync(
          inputText.slice(0, 30) + (inputText.length > 30 ? "..." : ""),
        )
        currentSessionId = newSession.id
        setActiveSessionId(newSession.id)
      } catch {
        return
      }
    }

    const queryToSend = inputText.trim()
    setInputText("")
    setIsStreaming(true)
    setStreamingContent("")
    setStreamingSources([])

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      await ChatService.streamSessionTokens(
        currentSessionId,
        queryToSend,
        (token) => {
          setStreamingContent((prev) => prev + token)
        },
        (sources) => {
          setStreamingSources(sources)
        },
        () => {
          setIsStreaming(false)
          setStreamingContent("")
          setStreamingSources([])
          queryClient.invalidateQueries({
            queryKey: ["chatSession", currentSessionId],
          })
          queryClient.invalidateQueries({ queryKey: ["aiUsage"] })
        },
        controller.signal,
      )
    } catch (err: unknown) {
      if ((err as Error)?.name !== "AbortError") {
        toast.error((err as Error)?.message || "Failed to stream chat response")
      }
      setIsStreaming(false)
      queryClient.invalidateQueries({
        queryKey: ["chatSession", currentSessionId],
      })
    } finally {
      abortControllerRef.current = null
    }
  }

  const toggleSources = (msgId: string) => {
    setExpandedSources((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }))
  }

  const messages: ChatMessagePublic[] = activeSession?.messages || []

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-primary" />
            Conversational RAG Chat
          </h1>
          <p className="text-muted-foreground text-sm">
            Multi-turn conversation with memory, hybrid retrieval, cross-encoder
            reranking, and live token streaming.
          </p>
        </div>
      </div>

      {/* Main Chat Workspace */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 flex-1 min-h-0">
        {/* Left Sidebar: Session List */}
        <Card className="md:col-span-1 flex flex-col h-full overflow-hidden">
          <CardHeader className="p-3 border-b flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm font-semibold flex items-center gap-1.5">
              <MessageSquare className="h-4 w-4" />
              Sessions ({sessions.length})
            </CardTitle>
            <Button
              size="sm"
              variant="outline"
              className="h-8 gap-1 text-xs"
              onClick={() => createSessionMutation.mutate(undefined)}
              disabled={createSessionMutation.isPending}
            >
              <Plus className="h-3.5 w-3.5" />
              New Chat
            </Button>
          </CardHeader>
          <CardContent className="p-2 flex-1 overflow-y-auto space-y-1">
            {isLoadingSessions ? (
              <div className="flex justify-center py-6">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-8 text-xs text-muted-foreground">
                No conversations yet.
                <br />
                Click "New Chat" to start!
              </div>
            ) : (
              sessions.map((sess: ChatSessionPublic) => {
                const isActive = sess.id === activeSessionId
                return (
                  <div
                    key={sess.id}
                    className={`group flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors ${
                      isActive
                        ? "bg-primary text-primary-foreground font-medium"
                        : "hover:bg-muted text-foreground"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => setActiveSessionId(sess.id)}
                      className="truncate flex-1 text-left px-1 py-1 text-xs outline-none focus-visible:underline"
                    >
                      {sess.title || "New Chat"}
                    </button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className={`h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity ${
                        isActive
                          ? "text-primary-foreground hover:bg-primary-foreground/20"
                          : "text-muted-foreground hover:text-destructive"
                      }`}
                      onClick={() => {
                        deleteSessionMutation.mutate(sess.id)
                      }}
                      title="Delete Conversation"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                )
              })
            )}
          </CardContent>
        </Card>

        {/* Right Area: Messages & Input */}
        <Card className="md:col-span-3 flex flex-col h-full overflow-hidden border">
          {/* Active Session Header Bar */}
          <div className="px-4 py-2.5 border-b flex items-center justify-between bg-muted/30">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm truncate">
                {activeSession?.title || "Select or start a chat"}
              </span>
              <Badge variant="outline" className="text-[10px] bg-background">
                gpt-4o-mini
              </Badge>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Database className="h-3.5 w-3.5 text-primary" />
              <span>pgvector hybrid + rerank</span>
            </div>
          </div>

          {/* Messages Scroll Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {isLoadingMessages ? (
              <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                <p className="text-xs">Loading conversation history...</p>
              </div>
            ) : messages.length === 0 && !isStreaming ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6 text-muted-foreground">
                <div className="rounded-full bg-primary/10 p-3 mb-3">
                  <Sparkles className="h-8 w-8 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground text-base">
                  How can I help you today?
                </h3>
                <p className="text-xs max-w-md mt-1 mb-4">
                  Ask any question. Answers are retrieved from your uploaded
                  documents via semantic vectors and cross-encoder re-ranking.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => {
                      setInputText("What is the return policy?")
                    }}
                    className="p-2.5 rounded-lg border bg-card hover:bg-muted text-left transition-colors"
                  >
                    "What is the return policy?"
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setInputText("How does the hybrid search work?")
                    }}
                    className="p-2.5 rounded-lg border bg-card hover:bg-muted text-left transition-colors"
                  >
                    "How does the hybrid search work?"
                  </button>
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg) => {
                  const isUser = msg.role === "user"
                  const hasSources = Boolean(
                    msg.sources && msg.sources.length > 0,
                  )
                  const isSourcesOpen = expandedSources[msg.id] ?? false

                  return (
                    <div
                      key={msg.id}
                      className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
                    >
                      {!isUser && (
                        <div className="h-8 w-8 rounded-full bg-primary/10 text-primary flex items-center justify-center shrink-0 mt-0.5">
                          <Bot className="h-4 w-4" />
                        </div>
                      )}

                      <div
                        className={`flex flex-col max-w-[85%] ${isUser ? "items-end" : "items-start"}`}
                      >
                        <div
                          className={`rounded-xl px-4 py-2.5 text-sm ${
                            isUser
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted/60 text-foreground border"
                          }`}
                        >
                          <div className="whitespace-pre-wrap leading-relaxed">
                            {msg.content}
                          </div>
                        </div>

                        {/* Citations / Sources Accordion */}
                        {!isUser && hasSources && (
                          <div className="mt-1.5 w-full">
                            <button
                              type="button"
                              onClick={() => toggleSources(msg.id)}
                              className="text-[11px] font-medium text-muted-foreground hover:text-primary flex items-center gap-1 transition-colors"
                            >
                              {isSourcesOpen ? (
                                <ChevronDown className="h-3 w-3" />
                              ) : (
                                <ChevronRight className="h-3 w-3" />
                              )}
                              <span>
                                Grounded in {msg.sources?.length} source chunk
                                {msg.sources && msg.sources.length > 1
                                  ? "s"
                                  : ""}
                              </span>
                            </button>

                            {isSourcesOpen && (
                              <div className="mt-1.5 space-y-1.5 pl-2 border-l-2 border-primary/30">
                                {msg.sources?.map((s, idx) => (
                                  <div
                                    key={`${s.chunk_id}-${idx}`}
                                    className="p-2 rounded bg-card/70 border text-xs space-y-0.5"
                                  >
                                    <div className="flex items-center justify-between font-medium">
                                      <span className="flex items-center gap-1 text-primary truncate">
                                        <FileText className="h-3 w-3" />
                                        {s.document_title} (Chunk #
                                        {s.chunk_index})
                                      </span>
                                      <Badge
                                        variant="outline"
                                        className="text-[10px] h-4"
                                      >
                                        {(s.score * 100).toFixed(0)}% match
                                      </Badge>
                                    </div>
                                    <span className="text-[10px] text-muted-foreground">
                                      Strategy: {s.match_type}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {isUser && (
                        <div className="h-8 w-8 rounded-full bg-secondary text-secondary-foreground flex items-center justify-center shrink-0 mt-0.5">
                          <UserIcon className="h-4 w-4" />
                        </div>
                      )}
                    </div>
                  )
                })}

                {/* Active Live Streaming Bubble */}
                {isStreaming && (
                  <div className="flex gap-3 justify-start">
                    <div className="h-8 w-8 rounded-full bg-primary/10 text-primary flex items-center justify-center shrink-0 mt-0.5">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="flex flex-col max-w-[85%] items-start">
                      <div className="rounded-xl px-4 py-2.5 text-sm bg-muted/60 text-foreground border">
                        <div className="whitespace-pre-wrap leading-relaxed">
                          {streamingContent}
                          <span className="inline-block w-1.5 h-4 ml-0.5 bg-primary animate-pulse align-middle" />
                        </div>
                      </div>

                      {streamingSources.length > 0 && (
                        <div className="mt-1.5 pl-2 text-[11px] text-muted-foreground flex items-center gap-1">
                          <Database className="h-3 w-3 text-primary animate-pulse" />
                          <span>
                            Streaming with {streamingSources.length} matched
                            context chunks
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Chat Input Bar */}
          <div className="p-3 border-t bg-background">
            <form onSubmit={handleSendMessage} className="flex gap-2 items-end">
              <Textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
                placeholder="Ask a question about your documents... (Enter to send, Shift+Enter for newline)"
                className="min-h-[44px] max-h-[120px] resize-none text-xs sm:text-sm py-2"
                disabled={isStreaming}
              />

              {isStreaming ? (
                <Button
                  type="button"
                  variant="destructive"
                  size="icon"
                  className="h-11 w-11 shrink-0"
                  onClick={handleStopStreaming}
                  title="Stop Generating"
                >
                  <Square className="h-4 w-4 fill-current" />
                </Button>
              ) : (
                <Button
                  type="submit"
                  size="icon"
                  className="h-11 w-11 shrink-0"
                  disabled={!inputText.trim()}
                >
                  <Send className="h-4 w-4" />
                </Button>
              )}
            </form>
          </div>
        </Card>
      </div>
    </div>
  )
}
