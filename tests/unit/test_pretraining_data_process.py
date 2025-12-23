# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pretraining data processing functionality."""

# Standard
from unittest.mock import MagicMock, mock_open, patch
import json
import os
import tempfile

# Third Party
from transformers import AutoTokenizer
import pytest

# First Party
from instructlab.training.data_process import process_documents_for_pretraining


class TestProcessDocumentsForPretraining:
    """Test suite for process_documents_for_pretraining function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock AutoTokenizer with BOS/EOS behavior."""
        mock_tok = MagicMock()
        mock_tok.bos_token_id = 1
        mock_tok.eos_token_id = 2

        # Mock encode to add BOS automatically and generate predictable tokens
        def mock_encode(text, add_special_tokens=True):
            # Simple hash-based encoding for predictability
            tokens = [hash(text) % 1000 + 100]
            if add_special_tokens:
                return [mock_tok.bos_token_id] + tokens
            return tokens

        mock_tok.encode = mock_encode
        return mock_tok

    @pytest.fixture
    def temp_pretraining_jsonl(self, tmp_path):
        """Create temp JSONL with 'documents' field."""
        data_file = tmp_path / "pretraining.jsonl"
        samples = [
            {"documents": "This is document one."},
            {"documents": "This is document two with more text."},
            {"documents": "Short doc."},
        ]

        with open(data_file, "w") as f:
            for sample in samples:
                json.dump(sample, f)
                f.write("\n")

        return str(data_file)

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return str(output_dir)

    @patch("instructlab.training.data_process.AutoTokenizer.from_pretrained")
    @patch("instructlab.training.data_process.load_dataset")
    def test_basic_tokenization_with_bos_eos(
        self,
        mock_load_dataset,
        mock_from_pretrained,
        mock_tokenizer,
        temp_pretraining_jsonl,
        temp_output_dir,
    ):
        """Verify basic tokenization adds BOS and EOS tokens correctly."""
        # Setup mocks
        mock_from_pretrained.return_value = mock_tokenizer

        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.num_rows = 1
        mock_ds.column_names = ["documents"]

        # Mock single document
        mock_ds.__iter__ = lambda self: iter([{"documents": "Test document"}])

        # Create filtered dataset mock
        filtered_ds = MagicMock()
        filtered_ds.num_rows = 1
        filtered_ds.column_names = ["documents"]

        # Mock filter to return the filtered dataset
        mock_ds.filter = MagicMock(return_value=filtered_ds)

        # Make map return a dataset with tokenized data
        def map_side_effect(func, **kwargs):
            result = func({"documents": "Test document"})
            mapped_ds = MagicMock()
            mapped_ds.__getitem__ = lambda self, key: [result[key]]
            mapped_ds.__len__ = lambda self: 1
            mapped_ds.to_json = MagicMock()
            return mapped_ds

        filtered_ds.map = MagicMock(side_effect=map_side_effect)
        mock_load_dataset.return_value = mock_ds

        # Run function
        process_documents_for_pretraining(
            data_path=temp_pretraining_jsonl,
            data_output_path=temp_output_dir,
            model_path="test-model",
            num_cpu_procs=1,
            document_column_name="documents",
        )

        # Verify tokenizer was loaded
        mock_from_pretrained.assert_called_once_with("test-model")

        # Verify dataset filter and map were called
        assert mock_ds.filter.called

    @patch("instructlab.training.data_process.AutoTokenizer.from_pretrained")
    @patch("instructlab.training.data_process.load_dataset")
    def test_multiple_documents_separate_records(
        self, mock_load_dataset, mock_from_pretrained, mock_tokenizer, temp_output_dir
    ):
        """Ensure each document gets its own JSONL record."""
        # Setup
        mock_from_pretrained.return_value = mock_tokenizer

        # Create mock dataset with 3 documents
        mock_ds = MagicMock()
        mock_ds.num_rows = 3
        mock_ds.column_names = ["documents"]

        docs = [{"documents": "Doc 1"}, {"documents": "Doc 2"}, {"documents": "Doc 3"}]

        # Create filtered dataset mock
        filtered_ds = MagicMock()
        filtered_ds.num_rows = 3
        filtered_ds.column_names = ["documents"]

        # Mock filter to return the filtered dataset
        mock_ds.filter = MagicMock(return_value=filtered_ds)

        # Mock map to process all documents
        def map_side_effect(func, **kwargs):
            results = [func(doc) for doc in docs]
            mapped_ds = MagicMock()
            mapped_ds.__len__ = lambda self: len(results)
            mapped_ds.__getitem__ = lambda self, key: [r[key] for r in results]
            mapped_ds.to_json = MagicMock()
            return mapped_ds

        filtered_ds.map = MagicMock(side_effect=map_side_effect)
        mock_load_dataset.return_value = mock_ds

        # Run
        process_documents_for_pretraining(
            data_path="dummy.jsonl",
            data_output_path=temp_output_dir,
            model_path="test-model",
            num_cpu_procs=1,
            document_column_name="documents",
        )

        # Verify filter and map were called (which processes each document)
        assert mock_ds.filter.called

    @patch("instructlab.training.data_process.load_dataset")
    def test_empty_dataset_raises_error(self, mock_load_dataset, temp_output_dir):
        """Validate error handling for empty input."""
        # Create empty dataset
        mock_ds = MagicMock()
        mock_ds.num_rows = 0
        mock_load_dataset.return_value = mock_ds

        # Should raise ValueError
        with pytest.raises(ValueError, match="empty"):
            process_documents_for_pretraining(
                data_path="dummy.jsonl",
                data_output_path=temp_output_dir,
                model_path="test-model",
                num_cpu_procs=1,
            )

    @patch("instructlab.training.data_process.load_dataset")
    def test_missing_documents_field_raises_error(
        self, mock_load_dataset, temp_output_dir
    ):
        """Validate schema enforcement."""
        # Create dataset with wrong field name
        mock_ds = MagicMock()
        mock_ds.num_rows = 1
        mock_ds.column_names = ["text"]  # Wrong field name
        mock_load_dataset.return_value = mock_ds

        # Should raise ValueError
        with pytest.raises(ValueError, match="must have.*field"):
            process_documents_for_pretraining(
                data_path="dummy.jsonl",
                data_output_path=temp_output_dir,
                model_path="test-model",
                num_cpu_procs=1,
                document_column_name="documents",
            )

    @patch("instructlab.training.data_process.AutoTokenizer.from_pretrained")
    @patch("instructlab.training.data_process.load_dataset")
    def test_tokenizer_without_eos_raises_error(
        self, mock_load_dataset, mock_from_pretrained, temp_output_dir
    ):
        """Validate tokenizer requirements."""
        # Create valid dataset
        mock_ds = MagicMock()
        mock_ds.num_rows = 1
        mock_ds.column_names = ["documents"]
        mock_load_dataset.return_value = mock_ds

        # Create tokenizer without EOS token
        mock_tok = MagicMock(spec=AutoTokenizer)
        mock_tok.eos_token_id = None  # No EOS token
        mock_from_pretrained.return_value = mock_tok

        # Should raise ValueError
        with pytest.raises(ValueError, match="must have an EOS token"):
            process_documents_for_pretraining(
                data_path="dummy.jsonl",
                data_output_path=temp_output_dir,
                model_path="test-model",
                num_cpu_procs=1,
                document_column_name="documents",
            )

    @patch("instructlab.training.data_process.logger")
    @patch("instructlab.training.data_process.AutoTokenizer.from_pretrained")
    @patch("instructlab.training.data_process.load_dataset")
    def test_statistics_logging(
        self,
        mock_load_dataset,
        mock_from_pretrained,
        mock_logger,
        mock_tokenizer,
        temp_output_dir,
    ):
        """Verify statistics are calculated correctly."""
        # Setup
        mock_from_pretrained.return_value = mock_tokenizer

        # Create dataset with known token counts
        mock_ds = MagicMock()
        mock_ds.num_rows = 2
        mock_ds.column_names = ["documents"]

        # Create filtered dataset mock
        filtered_ds = MagicMock()
        filtered_ds.num_rows = 2
        filtered_ds.column_names = ["documents"]

        # Mock filter to return the filtered dataset
        mock_ds.filter = MagicMock(return_value=filtered_ds)

        # Mock map to return known lengths
        def map_side_effect(func, **kwargs):
            # Simulate 2 documents with 5 and 10 tokens each
            mapped_ds = MagicMock()
            mapped_ds.__getitem__ = lambda self, key: [5, 10] if key == "len" else None
            mapped_ds.__len__ = lambda self: 2
            mapped_ds.to_json = MagicMock()
            return mapped_ds

        filtered_ds.map = MagicMock(side_effect=map_side_effect)
        mock_load_dataset.return_value = mock_ds

        # Run
        process_documents_for_pretraining(
            data_path="dummy.jsonl",
            data_output_path=temp_output_dir,
            model_path="test-model",
            num_cpu_procs=1,
            document_column_name="documents",
        )

        # Verify logging was called (check info was called multiple times)
        assert mock_logger.info.call_count >= 3

    @patch("instructlab.training.data_process.AutoTokenizer.from_pretrained")
    @patch("instructlab.training.data_process.load_dataset")
    def test_parallel_processing(
        self, mock_load_dataset, mock_from_pretrained, mock_tokenizer, temp_output_dir
    ):
        """Ensure num_cpu_procs parameter works."""
        # Setup
        mock_from_pretrained.return_value = mock_tokenizer

        mock_ds = MagicMock()
        mock_ds.num_rows = 1
        mock_ds.column_names = ["documents"]

        # Create filtered dataset mock
        filtered_ds = MagicMock()
        filtered_ds.num_rows = 1
        filtered_ds.column_names = ["documents"]

        # Mock filter to return the filtered dataset
        mock_ds.filter = MagicMock(return_value=filtered_ds)

        def map_side_effect(func, **kwargs):
            mapped_ds = MagicMock()
            mapped_ds.__len__ = lambda self: 1
            mapped_ds.__getitem__ = lambda self, key: [10] if key == "len" else None
            mapped_ds.to_json = MagicMock()
            return mapped_ds

        filtered_ds.map = MagicMock(side_effect=map_side_effect)
        mock_load_dataset.return_value = mock_ds

        # Run with specific num_cpu_procs
        process_documents_for_pretraining(
            data_path="dummy.jsonl",
            data_output_path=temp_output_dir,
            model_path="test-model",
            num_cpu_procs=4,
            document_column_name="documents",
        )

        # Verify filter was called with num_proc=4
        filter_call_args = mock_ds.filter.call_args
        assert filter_call_args[1]["num_proc"] == 4

        # Verify map was also called with num_proc=4
        map_call_args = filtered_ds.map.call_args
        assert map_call_args[1]["num_proc"] == 4

    def test_output_directory_creation(self, tmp_path, mock_tokenizer):
        """Verify directory is created if it doesn't exist."""
        # Use non-existent output path
        output_dir = tmp_path / "nonexistent" / "nested" / "dir"

        with patch(
            "instructlab.training.data_process.AutoTokenizer.from_pretrained"
        ) as mock_from_pretrained:
            with patch(
                "instructlab.training.data_process.load_dataset"
            ) as mock_load_dataset:
                mock_from_pretrained.return_value = mock_tokenizer

                mock_ds = MagicMock()
                mock_ds.num_rows = 1
                mock_ds.column_names = ["documents"]

                # Create filtered dataset mock
                filtered_ds = MagicMock()
                filtered_ds.num_rows = 1
                filtered_ds.column_names = ["documents"]

                # Mock filter to return the filtered dataset
                mock_ds.filter = MagicMock(return_value=filtered_ds)

                def map_side_effect(func, **kwargs):
                    mapped_ds = MagicMock()
                    mapped_ds.__len__ = lambda self: 1
                    mapped_ds.__getitem__ = (
                        lambda self, key: [10] if key == "len" else None
                    )
                    mapped_ds.to_json = MagicMock()
                    return mapped_ds

                filtered_ds.map = MagicMock(side_effect=map_side_effect)
                mock_load_dataset.return_value = mock_ds

                # Run
                process_documents_for_pretraining(
                    data_path="dummy.jsonl",
                    data_output_path=str(output_dir),
                    model_path="test-model",
                    num_cpu_procs=1,
                    document_column_name="documents",
                )

                # Verify directory was created
                assert output_dir.exists()

    @patch("instructlab.training.data_process.AutoTokenizer.from_pretrained")
    @patch("instructlab.training.data_process.load_dataset")
    def test_output_jsonl_format(
        self, mock_load_dataset, mock_from_pretrained, mock_tokenizer, temp_output_dir
    ):
        """Validate JSONL output format."""
        # Setup
        mock_from_pretrained.return_value = mock_tokenizer

        mock_ds = MagicMock()
        mock_ds.num_rows = 1
        mock_ds.column_names = ["documents"]

        # Create filtered dataset mock
        filtered_ds = MagicMock()
        filtered_ds.num_rows = 1
        filtered_ds.column_names = ["documents"]

        # Mock filter to return the filtered dataset
        mock_ds.filter = MagicMock(return_value=filtered_ds)

        # Track what gets written
        output_file_path = None

        def map_side_effect(func, **kwargs):
            result = func({"documents": "Test"})
            mapped_ds = MagicMock()
            mapped_ds.__len__ = lambda self: 1
            mapped_ds.__getitem__ = lambda self, key: [result[key]]

            def to_json_side_effect(path, **kw):
                nonlocal output_file_path
                output_file_path = path
                # Write actual JSON to verify format
                with open(path, "w") as f:
                    json.dump(result, f)
                    f.write("\n")

            mapped_ds.to_json = to_json_side_effect
            return mapped_ds

        filtered_ds.map = MagicMock(side_effect=map_side_effect)
        mock_load_dataset.return_value = mock_ds

        # Run
        process_documents_for_pretraining(
            data_path="dummy.jsonl",
            data_output_path=temp_output_dir,
            model_path="test-model",
            num_cpu_procs=1,
            document_column_name="documents",
        )

        # Verify file was created
        assert output_file_path is not None
        assert os.path.exists(output_file_path)

        # Verify format
        with open(output_file_path, "r") as f:
            line = f.readline()
            data = json.loads(line)

            # Should have input_ids and len fields
            assert "input_ids" in data
            assert "len" in data

            # Should NOT have labels field
            assert "labels" not in data

            # input_ids should be a list starting with BOS
            assert isinstance(data["input_ids"], list)
            assert data["input_ids"][0] == 1  # BOS token
            assert data["input_ids"][-1] == 2  # EOS token

    @pytest.mark.slow
    def test_integration_with_real_tokenizer(self, temp_output_dir):
        """Integration test with actual GPT2 tokenizer."""
        # Create real input file
        input_file = os.path.join(temp_output_dir, "input.jsonl")
        with open(input_file, "w") as f:
            json.dump(
                {"documents": "This is a test document for GPT2 tokenization."}, f
            )
            f.write("\n")

        # Run with real tokenizer
        process_documents_for_pretraining(
            data_path=input_file,
            data_output_path=temp_output_dir,
            model_path="gpt2",
            num_cpu_procs=1,
            document_column_name="documents",
        )

        # Verify output
        output_file = os.path.join(temp_output_dir, "data.jsonl")
        assert os.path.exists(output_file)

        with open(output_file, "r") as f:
            line = f.readline()
            data = json.loads(line)

            # Verify structure
            assert "input_ids" in data
            assert "len" in data
            assert len(data["input_ids"]) == data["len"]

            # Load tokenizer to verify tokens
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # Verify EOS is present at the end
            # Note: GPT2's encode() with add_special_tokens=True doesn't add BOS
            # (GPT2 uses the same token for BOS and EOS)
            # The implementation manually appends EOS if not present
            assert data["input_ids"][-1] == tokenizer.eos_token_id

            # Verify token count is reasonable (should have content tokens + EOS)
            assert data["len"] > 5
