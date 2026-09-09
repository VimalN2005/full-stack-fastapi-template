import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  Coins,
  FileText,
  Loader2,
  Plus,
  RefreshCw,
  Trash2,
  Zap,
} from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import {
  type DocumentCreate,
  type DocumentPublic,
  RagService,
} from "@/services/ai"

export const Route = createFileRoute("/_layout/documents")({
  component: DocumentsPage,
  head: () => ({
    meta: [{ title: "Knowledge Base - FastAPI Template" }],
  }),
})

const SAMPLE_DOCS = [
  {
    title: "Company Return Policy",
    type: "text/plain",
    content:
      "Return and Refund Policy:\nCustomers can return items within 30 days of purchase for a full refund. Items must be in original condition with tags intact. Electronics have a 14-day return window. Return shipping is free for loyalty club members.",
  },
  {
    title: "Deployment & Architecture Guide",
    type: "text/markdown",
    content:
      "# System Architecture\nThe backend is powered by FastAPI and PostgreSQL with pgvector for semantic vector embeddings. Multi-tenant isolation ensures each user's vector embeddings are strictly partitioned. Streaming endpoints provide real-time token generation with client-disconnect detection.",
  },
]

function DocumentsPage() {
  const queryClient = useQueryClient()
  const [isOpen, setIsOpen] = useState(false)
  const [title, setTitle] = useState("")
  const [content, setContent] = useState("")
  const [contentType, setContentType] = useState("text/plain")
  const [background, setBackground] = useState(true)

  // AI Usage Query
  const { data: aiUsage, refetch: refetchUsage } = useQuery({
    queryKey: ["aiUsage"],
    queryFn: () => RagService.getAIUsage(),
    staleTime: 10_000,
  })

  // Documents Query with dynamic polling when items are processing
  const {
    data: documentsData,
    isLoading: isLoadingDocs,
    refetch: refetchDocs,
    isRefetching,
  } = useQuery({
    queryKey: ["ragDocuments"],
    queryFn: () => RagService.getDocuments(0, 100),
    refetchInterval: (query) => {
      const data = query.state.data
      const hasProcessing = data?.data?.some(
        (d: DocumentPublic) => d.status === "processing",
      )
      return hasProcessing ? 2000 : false
    },
  })

  // Add Document Mutation
  const createDocMutation = useMutation({
    mutationFn: (newDoc: { data: DocumentCreate; background: boolean }) =>
      RagService.createDocument(newDoc.data, newDoc.background),
    onSuccess: (createdDoc) => {
      if (background) {
        toast.success(
          "Document queued for processing! Background task is embedding chunks.",
        )
      } else {
        toast.success(
          `Document '${createdDoc.title}' embedded successfully (${createdDoc.chunk_count} chunks).`,
        )
      }
      queryClient.invalidateQueries({ queryKey: ["ragDocuments"] })
      queryClient.invalidateQueries({ queryKey: ["aiUsage"] })
      setIsOpen(false)
      setTitle("")
      setContent("")
    },
    onError: (error: Error) => {
      toast.error(error.message || "Failed to create document")
    },
  })

  // Delete Document Mutation
  const deleteDocMutation = useMutation({
    mutationFn: (id: string) => RagService.deleteDocument(id),
    onSuccess: () => {
      toast.success("Document and its vector embeddings deleted")
      queryClient.invalidateQueries({ queryKey: ["ragDocuments"] })
    },
    onError: (error: Error) => {
      toast.error(error.message || "Failed to delete document")
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || !content.trim()) {
      toast.error("Please fill in both title and content")
      return
    }
    createDocMutation.mutate({
      data: {
        title: title.trim(),
        content: content.trim(),
        content_type: contentType,
      },
      background,
    })
  }

  const loadSample = (sample: (typeof SAMPLE_DOCS)[0]) => {
    setTitle(sample.title)
    setContentType(sample.type)
    setContent(sample.content)
  }

  const documents = documentsData?.data || []
  const costPct = aiUsage?.budget_limit_usd
    ? Math.min(
        100,
        Math.round(
          (aiUsage.estimated_cost_usd / aiUsage.budget_limit_usd) * 100,
        ),
      )
    : 0

  return (
    <div className="flex flex-col gap-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <BookOpen className="h-6 w-6 text-primary" />
            Knowledge Base & RAG Documents
          </h1>
          <p className="text-muted-foreground">
            Upload text documents for hybrid semantic retrieval, pgvector
            indexing, and AI grounding.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              refetchDocs()
              refetchUsage()
            }}
            disabled={isRefetching}
            className="flex items-center gap-1.5"
          >
            <RefreshCw
              className={`h-4 w-4 ${isRefetching ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>

          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button className="flex items-center gap-1.5">
                <Plus className="h-4 w-4" />
                Add Document
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[600px]">
              <form onSubmit={handleSubmit}>
                <DialogHeader>
                  <DialogTitle>Add New Knowledge Document</DialogTitle>
                  <DialogDescription>
                    The document will be tokenized, chunked with sliding
                    overlap, and embedded into pgvector.
                  </DialogDescription>
                </DialogHeader>

                <div className="flex items-center gap-2 my-3 text-xs text-muted-foreground">
                  <span>Quick Templates:</span>
                  {SAMPLE_DOCS.map((sample) => (
                    <Button
                      key={sample.title}
                      type="button"
                      variant="secondary"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => loadSample(sample)}
                    >
                      {sample.title}
                    </Button>
                  ))}
                </div>

                <div className="grid gap-4 py-2">
                  <div className="grid gap-2">
                    <Label htmlFor="doc-title">Document Title</Label>
                    <Input
                      id="doc-title"
                      placeholder="e.g. API Architecture or FAQ"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      required
                    />
                  </div>

                  <div className="grid gap-2">
                    <Label htmlFor="doc-type">Content Type</Label>
                    <Input
                      id="doc-type"
                      placeholder="text/plain or text/markdown"
                      value={contentType}
                      onChange={(e) => setContentType(e.target.value)}
                    />
                  </div>

                  <div className="grid gap-2">
                    <Label htmlFor="doc-content">Document Content</Label>
                    <Textarea
                      id="doc-content"
                      placeholder="Paste your raw text, knowledge article, policy, or markdown here..."
                      className="min-h-[160px] font-mono text-xs"
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      required
                    />
                  </div>

                  <div className="flex items-center space-x-2 pt-1">
                    <Checkbox
                      id="background"
                      checked={background}
                      onCheckedChange={(c) => setBackground(Boolean(c))}
                    />
                    <div className="grid gap-0.5 leading-none">
                      <label
                        htmlFor="background"
                        className="text-sm font-medium cursor-pointer"
                      >
                        Process in Background Task (Recommended)
                      </label>
                      <p className="text-xs text-muted-foreground">
                        Dispatches chunking and pgvector embedding
                        asynchronously without blocking the UI.
                      </p>
                    </div>
                  </div>
                </div>

                <DialogFooter className="mt-4">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setIsOpen(false)}
                    disabled={createDocMutation.isPending}
                  >
                    Cancel
                  </Button>
                  <Button type="submit" disabled={createDocMutation.isPending}>
                    {createDocMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        {background ? "Queuing..." : "Embedding..."}
                      </>
                    ) : (
                      "Ingest Document"
                    )}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* AI Token & Quota Metrics Card */}
      {aiUsage && (
        <Card className="bg-card/50 border-primary/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Coins className="h-5 w-5 text-amber-500" />
                <CardTitle className="text-base">
                  Monthly AI Token & Cost Metering
                </CardTitle>
              </div>
              <Badge
                variant={
                  aiUsage.is_budget_exceeded ? "destructive" : "secondary"
                }
                className="font-mono text-xs"
              >
                {aiUsage.is_budget_exceeded
                  ? "Quota Exceeded (HTTP 429)"
                  : "Quota Active"}
              </Badge>
            </div>
            <CardDescription>
              Real-time token tracking and cost guardrails for current billing
              cycle ({aiUsage.month}).
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-1">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Total Tokens</p>
                <p className="text-xl font-bold font-mono">
                  {aiUsage.total_tokens.toLocaleString()}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  {aiUsage.prompt_tokens.toLocaleString()} prompt /{" "}
                  {aiUsage.completion_tokens.toLocaleString()} comp
                </p>
              </div>

              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Estimated Spend</p>
                <p className="text-xl font-bold font-mono text-emerald-500">
                  ${aiUsage.estimated_cost_usd.toFixed(4)}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  Budget limit: ${aiUsage.budget_limit_usd.toFixed(2)}
                </p>
              </div>

              <div className="space-y-1 col-span-2">
                <div className="flex justify-between text-xs text-muted-foreground mb-1.5">
                  <span>Budget Utilization</span>
                  <span className="font-mono">{costPct}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      aiUsage.is_budget_exceeded
                        ? "bg-destructive"
                        : costPct > 80
                          ? "bg-amber-500"
                          : "bg-primary"
                    }`}
                    style={{ width: `${Math.max(2, Math.min(100, costPct))}%` }}
                  />
                </div>
                <p className="text-[11px] text-muted-foreground flex items-center gap-1 pt-1">
                  <Zap className="h-3 w-3 text-amber-400" />
                  Hard guardrail prevents unexpected cloud LLM overspending.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Documents Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <FileText className="h-5 w-5 text-muted-foreground" />
            Ingested Documents ({documents.length})
          </CardTitle>
          <CardDescription>
            Documents are automatically broken into overlapping chunks and
            searchable via pgvector hybrid search.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          {isLoadingDocs ? (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">
                Loading documents...
              </p>
            </div>
          ) : documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="rounded-full bg-muted p-4 mb-3">
                <BookOpen className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="text-base font-semibold">
                No documents added yet
              </h3>
              <p className="text-sm text-muted-foreground max-w-sm mb-4">
                Add your first document to enable grounded RAG answering in the
                AI Chat.
              </p>
              <Button onClick={() => setIsOpen(true)} size="sm">
                <Plus className="mr-1.5 h-4 w-4" />
                Add Your First Document
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Title</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Chunks</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {documents.map((doc) => {
                  return (
                    <TableRow key={doc.id}>
                      <TableCell className="font-medium max-w-[240px] truncate">
                        {doc.title}
                        {doc.error_message && (
                          <p
                            className="text-xs text-destructive truncate"
                            title={doc.error_message}
                          >
                            {doc.error_message}
                          </p>
                        )}
                      </TableCell>

                      <TableCell>
                        {doc.status === "processing" ? (
                          <Badge
                            variant="outline"
                            className="bg-amber-500/10 text-amber-500 border-amber-500/30 flex items-center gap-1 w-fit"
                          >
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Processing...
                          </Badge>
                        ) : doc.status === "failed" ? (
                          <Badge
                            variant="destructive"
                            className="flex items-center gap-1 w-fit"
                          >
                            <AlertCircle className="h-3 w-3" />
                            Failed
                          </Badge>
                        ) : (
                          <Badge
                            variant="outline"
                            className="bg-emerald-500/10 text-emerald-500 border-emerald-500/30 flex items-center gap-1 w-fit"
                          >
                            <CheckCircle2 className="h-3 w-3" />
                            Ready
                          </Badge>
                        )}
                      </TableCell>

                      <TableCell className="text-muted-foreground text-xs font-mono">
                        {doc.content_type}
                      </TableCell>

                      <TableCell className="font-mono text-sm">
                        {doc.status === "processing" ? (
                          <span className="text-muted-foreground italic">
                            generating...
                          </span>
                        ) : (
                          `${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}`
                        )}
                      </TableCell>

                      <TableCell className="text-muted-foreground text-xs">
                        {new Date(doc.created_at).toLocaleDateString(
                          undefined,
                          {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                          },
                        )}
                      </TableCell>

                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="text-muted-foreground hover:text-destructive h-8 w-8"
                          onClick={() => {
                            if (
                              confirm(
                                `Delete document '${doc.title}' and all vector embeddings?`,
                              )
                            ) {
                              deleteDocMutation.mutate(doc.id)
                            }
                          }}
                          disabled={deleteDocMutation.isPending}
                          title="Delete Document"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
