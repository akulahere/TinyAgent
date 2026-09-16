import io
import json
import unittest
from unittest.mock import Mock, patch

from agent import TinyAgent
from llm import EmbeddingModel, LLM, Response
from memory import RAGMemory


class EmbeddingTests(unittest.TestCase):
    def test_posts_input_and_extracts_embedding(self):
        payload = {"data": [{"embedding": [0.25, -0.75]}]}
        with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as send:
            vector = EmbeddingModel("embedding-test", "http://localhost:11434/v1/", "test-key").embed("Birds")
        request = send.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "http://localhost:11434/v1/embeddings")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), {"model": "embedding-test", "input": "Birds"})
        self.assertEqual(vector, [0.25, -0.75])


class RAGTests(unittest.TestCase):
    def test_ranks_by_cosine_not_vector_magnitude(self):
        embeddings = Mock(spec=EmbeddingModel)
        embeddings.embed.side_effect = [[1, 0], [100, 100], [-1, 0], [0, 1], [1, 0]]
        memory = RAGMemory(embeddings, ["exact", "large but less similar", "opposite", "orthogonal"])
        self.assertEqual(memory.search("query"), ["exact", "large but less similar", "orthogonal"])

    def test_retrieval_is_sent_to_llm_and_original_query_is_in_trajectory(self):
        embeddings = Mock(spec=EmbeddingModel)
        embeddings.embed.side_effect = [[0, 1], [1, 0], [1, 0]]
        documents = ["Ilse likes dolphins.", "Sarah likes flamingos."]
        memory = RAGMemory(embeddings, documents, top_k=1)
        documents.append("External mutation")
        llm = Mock(spec=LLM)
        llm.generate.return_value = Response("Flamingos", None, None, {})
        agent = TinyAgent(llm, memory)
        query = "Which animal does Sarah like?"
        self.assertEqual(agent.run(query), "Flamingos")
        prompt = llm.generate.call_args.args[0][0]["content"]
        self.assertIn("Context:\nSarah likes flamingos.", prompt)
        self.assertIn("Question: " + query, prompt)
        self.assertNotIn("dolphins", prompt)
        self.assertEqual(agent.trajectory.runs[0]["query"], query)
        self.assertEqual(memory.get_messages()[-1], {"role": "assistant", "content": "Flamingos"})
        self.assertEqual(embeddings.embed.call_count, 3)

    def test_empty_documents_need_no_embedding_requests(self):
        embeddings = Mock(spec=EmbeddingModel)
        memory = RAGMemory(embeddings, [])
        self.assertEqual(memory.search("query"), [])
        memory.add("user", "query")
        self.assertEqual(memory.get_messages()[0]["content"], "query")
        embeddings.embed.assert_not_called()

    def test_cosine_handles_zero_vectors_and_rejects_dimension_mismatch(self):
        self.assertEqual(RAGMemory._cosine([0, 0], [1, 0]), 0)
        self.assertAlmostEqual(RAGMemory._cosine([1, 0], [-1, 0]), -1)
        with self.assertRaises(ValueError):
            RAGMemory._cosine([1], [1, 2])

    def test_nonpositive_top_k_is_rejected_before_embedding(self):
        embeddings = Mock(spec=EmbeddingModel)
        with self.assertRaises(ValueError):
            RAGMemory(embeddings, ["doc"], top_k=0)
        embeddings.embed.assert_not_called()
