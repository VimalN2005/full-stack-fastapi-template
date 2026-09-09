import { client } from "@/client/client.gen"

export interface DocumentPublic {
  id: string
  user_id: string
  title: string
  content_type: string
  chunk_count: number
  status: "processing" | "ready" | "failed"
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface DocumentStatusResponse {
  id: string
  title: string
  status: "processing" | "ready" | "failed"
  chunk_count: number
  error_message: string | null
  updated_at: string
}

export interface DocumentsPublic {
  data: DocumentPublic[]
  count: number
}

export interface DocumentCreate {
  title: string
  content: string
  content_type?: string
}

export interface UserAIUsagePublic {
  user_id: string
  month: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  estimated_cost_usd: number
  budget_limit_usd: number
  is_budget_exceeded: boolean
}

export interface ChatSource {
  chunk_id: string
  document_id: string
  document_title: string
  chunk_index: number
  score: number
  match_type: string
}

export interface ChatMessagePublic {
  id: string
  session_id: string
  role: "user" | "assistant" | "system"
  content: string
  sources?: ChatSource[] | null
  created_at: string
}

export interface ChatSessionPublic {
  id: string
  user_id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ChatSessionDetailPublic extends ChatSessionPublic {
  messages: ChatMessagePublic[]
}

export interface ChatSessionsPublic {
  data: ChatSessionPublic[]
  count: number
}

export const RagService = {
  async getDocuments(skip = 0, limit = 100): Promise<DocumentsPublic> {
    const res = await client.get<{ data: DocumentsPublic }>({
      url: "/api/v1/rag/documents",
      query: { skip, limit },
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as DocumentsPublic
  },

  async createDocument(
    data: DocumentCreate,
    background = false,
  ): Promise<DocumentPublic> {
    const res = await client.post<{ data: DocumentPublic }>({
      url: `/api/v1/rag/documents${background ? "?background=true" : ""}`,
      body: data,
      headers: { "Content-Type": "application/json" },
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as DocumentPublic
  },

  async getDocumentStatus(id: string): Promise<DocumentStatusResponse> {
    const res = await client.get<{ data: DocumentStatusResponse }>({
      url: `/api/v1/rag/documents/${id}/status`,
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as DocumentStatusResponse
  },

  async deleteDocument(id: string): Promise<{ message: string }> {
    const res = await client.delete<{ data: { message: string } }>({
      url: `/api/v1/rag/documents/${id}`,
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as { message: string }
  },

  async getAIUsage(): Promise<UserAIUsagePublic> {
    const res = await client.get<{ data: UserAIUsagePublic }>({
      url: "/api/v1/ai/usage",
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as UserAIUsagePublic
  },
}

export const ChatService = {
  async getSessions(skip = 0, limit = 50): Promise<ChatSessionsPublic> {
    const res = await client.get<{ data: ChatSessionsPublic }>({
      url: "/api/v1/chat/sessions",
      query: { skip, limit },
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as ChatSessionsPublic
  },

  async createSession(title?: string): Promise<ChatSessionPublic> {
    const res = await client.post<{ data: ChatSessionPublic }>({
      url: "/api/v1/chat/sessions",
      body: { title: title || "New Chat" },
      headers: { "Content-Type": "application/json" },
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as ChatSessionPublic
  },

  async getSession(id: string): Promise<ChatSessionDetailPublic> {
    const res = await client.get<{ data: ChatSessionDetailPublic }>({
      url: `/api/v1/chat/sessions/${id}`,
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as ChatSessionDetailPublic
  },

  async deleteSession(id: string): Promise<{ message: string }> {
    const res = await client.delete<{ data: { message: string } }>({
      url: `/api/v1/chat/sessions/${id}`,
      security: [{ scheme: "bearer", type: "http" }],
    })
    return res.data as { message: string }
  },

  async streamSessionTokens(
    sessionId: string,
    content: string,
    onToken: (token: string) => void,
    onSources: (sources: ChatSource[]) => void,
    onDone: (meta: { status: string; total_tokens: number }) => void,
    signal?: AbortSignal,
    top_k = 5,
    rerank = true,
  ): Promise<void> {
    const baseUrl = import.meta.env.VITE_API_URL ?? ""
    const token = localStorage.getItem("access_token") || ""

    const response = await fetch(
      `${baseUrl}/api/v1/chat/sessions/${sessionId}/stream`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ content, top_k, rerank }),
        signal,
      },
    )

    if (!response.ok) {
      const errorText = await response.text()
      let detail = "Error streaming chat response"
      try {
        const errorJson = JSON.parse(errorText)
        detail = errorJson.detail || detail
      } catch {
        // use raw text
      }
      throw new Error(detail)
    }

    if (!response.body) {
      throw new Error("ReadableStream not supported on response body")
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split("\n\n")
      buffer = parts.pop() || ""

      for (const part of parts) {
        if (!part.trim()) continue
        const lines = part.split("\n")
        let currentEvent = ""
        let currentData = ""

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.substring(7).trim()
          } else if (line.startsWith("data: ")) {
            currentData = line.substring(6).trim()
          }
        }

        if (currentEvent === "token") {
          onToken(currentData)
        } else if (currentEvent === "sources") {
          try {
            const parsed = JSON.parse(currentData)
            onSources(parsed)
          } catch (e) {
            console.error("Failed to parse sources SSE data:", e)
          }
        } else if (currentEvent === "done") {
          try {
            const parsed = JSON.parse(currentData)
            onDone(parsed)
          } catch {
            onDone({ status: "completed", total_tokens: 0 })
          }
        } else if (currentEvent === "error") {
          try {
            const parsed = JSON.parse(currentData)
            throw new Error(parsed.detail || "Streaming error")
          } catch {
            throw new Error(currentData || "Streaming error")
          }
        }
      }
    }
  },
}
