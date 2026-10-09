"""Our PDF loader: one Document per page, with the metadata search results cite."""

from app.pdf_loader import load_pdf, load_pdf_folder


def test_one_document_per_page_with_citation_metadata(tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "paper.pdf", pages=3)
    docs = load_pdf(pdf)

    assert len(docs) == 3
    assert [d.metadata["page"] for d in docs] == [0, 1, 2]           # 0-based, like pypdf
    assert [d.metadata["page_label"] for d in docs] == ["1", "2", "3"]
    assert all(d.metadata["source"] == str(pdf) for d in docs)
    assert all(d.metadata["total_pages"] == 3 for d in docs)


def test_folder_loads_pdfs_in_order_and_skips_hidden_and_other_files(tmp_path, make_pdf):
    make_pdf(tmp_path / "b.pdf", pages=1)
    make_pdf(tmp_path / "a.pdf", pages=2)
    make_pdf(tmp_path / ".hidden.pdf", pages=1)
    (tmp_path / "notes.txt").write_text("not a pdf")

    sources = [d.metadata["source"] for d in load_pdf_folder(tmp_path)]
    assert sources == [str(tmp_path / "a.pdf")] * 2 + [str(tmp_path / "b.pdf")]
