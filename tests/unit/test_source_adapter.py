import httpx

from nlp_academic_search.data.sources.arxiv_oai import DEFAULT_ARXIV_OAI_ENDPOINT, ArxivOAIAdapter

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords><record><header><identifier>oai:arXiv.org:1706.03762</identifier></header>
  <metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/">
    <id>1706.03762</id><created>2017-06-12</created><updated>2017-06-12</updated>
    <authors><author><keyname>Vaswani</keyname><forenames>Ashish</forenames></author></authors>
    <title>Attention Is All You Need</title><categories>cs.CL cs.LG</categories>
    <abstract>A Transformer based solely on attention.</abstract>
  </arXiv></metadata></record><resumptionToken></resumptionToken></ListRecords>
</OAI-PMH>"""


def test_arxiv_oai_adapter_parses_real_metadata_shape():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=XML))
    )
    adapter = ArxivOAIAdapter(client=client, request_interval_seconds=0)
    papers = list(adapter.iter_papers(max_records=1))
    assert papers[0].arxiv_id == "1706.03762"
    assert papers[0].authors == ["Ashish Vaswani"]
    assert papers[0].categories == ["cs.CL", "cs.LG"]


def test_arxiv_oai_adapter_uses_current_endpoint():
    adapter = ArxivOAIAdapter(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=XML))
        )
    )
    assert adapter.endpoint == DEFAULT_ARXIV_OAI_ENDPOINT


def test_arxiv_oai_adapter_can_follow_endpoint_redirects():
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "export.arxiv.org":
            return httpx.Response(
                301,
                headers={
                    "Location": str(request.url.copy_with(host="oaipmh.arxiv.org", path="/oai"))
                },
            )
        return httpx.Response(200, content=XML)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    adapter = ArxivOAIAdapter(
        endpoint="https://export.arxiv.org/oai2",
        client=client,
        request_interval_seconds=0,
    )
    papers = list(adapter.iter_papers(max_records=1))
    assert papers[0].arxiv_id == "1706.03762"
    assert requested_hosts == ["export.arxiv.org", "oaipmh.arxiv.org"]
