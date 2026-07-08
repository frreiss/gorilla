import os
import random
import time
from typing import Optional
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

import httpx
import re

import tavily
from .snippet import shorten_snippet

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


def _duckduckgo_region_code_to_tavily_country_code(region: Optional[str]) -> Optional[str]:
    """Map region codes from the format of the SerpAPI DuckDuckGo API to Tavily's format.
    
    :param region: A SerpAPI region code, suitable to pass to the ``kl`` paramter of
        the SerpAPI DuckDuckGo search API. 
        See https://serpapi.com/duckduckgo-search-api#api-parameters-localization
        for information about the ``kl`` parameter.
    
    :returns: The closest equivalent region string to pass in the ``country`` parameter
        of the Tavily search API. 
        See https://docs.tavily.com/documentation/api-reference/endpoint/search#body-country
        for information about the ``country`` parameter.
        Returns ``None`` if the ``country`` paramter should be omitted from the 
        equivalent Tavily API call.
    """
    # Maps SerpAPI DuckDuckGo region codes to Tavily's proprietary country name strings.
    # Codes with no single-country equivalent (wt-wt, xl-es, ct-ca, hk-tzh) map to None,
    # which callers should omit from the Tavily request entirely.
    _REGION_MAP = {
        "xa-ar": "saudi arabia",     # Arabia
        "xa-en": "saudi arabia",     # Arabia (en)
        "ar-es": "argentina",
        "au-en": "australia",
        "at-de": "austria",
        "be-fr": "belgium",
        "be-nl": "belgium",
        "br-pt": "brazil",
        "bg-bg": "bulgaria",
        "ca-en": "canada",
        "ca-fr": "canada",
        "ct-ca": None,               # Catalan — no single country equivalent
        "cl-es": "chile",
        "cn-zh": "china",
        "co-es": "colombia",
        "hr-hr": "croatia",
        "cz-cs": "czech republic",
        "dk-da": "denmark",
        "ee-et": "estonia",
        "fi-fi": "finland",
        "fr-fr": "france",
        "de-de": "germany",
        "gr-el": "greece",
        "hk-tzh": None,              # Hong Kong — not in Tavily's enumeration
        "hu-hu": "hungary",
        "in-en": "india",
        "id-id": "indonesia",
        "id-en": "indonesia",
        "ie-en": "ireland",
        "il-he": "israel",
        "it-it": "italy",
        "jp-jp": "japan",
        "kr-kr": "south korea",
        "lv-lv": "latvia",
        "lt-lt": "lithuania",
        "xl-es": None,               # Latin America — no single country equivalent
        "my-ms": "malaysia",
        "my-en": "malaysia",
        "mx-es": "mexico",
        "nl-nl": "netherlands",
        "nz-en": "new zealand",
        "no-no": "norway",
        "pe-es": "peru",
        "ph-en": "philippines",
        "ph-tl": "philippines",
        "pl-pl": "poland",
        "pt-pt": "portugal",
        "ro-ro": "romania",
        "ru-ru": "russia",
        "sg-en": "singapore",
        "sk-sk": "slovakia",
        "sl-sl": "slovenia",
        "za-en": "south africa",
        "es-es": "spain",
        "se-sv": "sweden",
        "ch-de": "switzerland",
        "ch-fr": "switzerland",
        "ch-it": "switzerland",
        "tw-tzh": "taiwan",
        "th-th": "thailand",
        "tr-tr": "turkey",
        "ua-uk": "ukraine",
        "uk-en": "united kingdom",
        "us-en": "united states",
        "ue-es": "united states",
        "ve-es": "venezuela",
        "vn-vi": "vietnam",
        "wt-wt": None,               # No region
    }
    return _REGION_MAP.get(region)



class WebSearchAPI:
    def __init__(self):
        self._api_description = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results."
        self.show_snippet = True
        # Note: The following two random generators are used to simulate random errors, but that feature is not currently used
        # This one used to determine if we should simulate a random error
        # Outcome (True means simulate error): [True, False, True, True, False, True, True, True, False, False, True, True, False, True, False, False, False, False, False, True]
        self._random = random.Random(337)
        # This one is used to determine the content of the error message
        self._rng = random.Random(1053)

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # We don't care about the long_context parameter here
        # It's there to match the signature of functions in the multi-turn evaluation code
        self.show_snippet = initial_config["show_snippet"]

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """Redirect to appropriate implementation. See
        search_engine_query_original() for full docs.
        """
        # Allow switching engines based on an environment variable.
        # See code here for possible values and defaults.
        search_engine_name = os.environ.get("SEARCH_ENGINE_NAME", "Tavily")
        #print(f"Using search engine '{search_engine_name}'")
        if search_engine_name == "Tavily":
            return self.search_with_tavily(keywords, max_results, region)
        if search_engine_name == "MCP":
            #print("Using MCP search")
            return self.search_with_ibm_mcp(keywords, max_results)
        if search_engine_name == "SerpAPI":
            return self.search_engine_query_original(keywords, max_results, region)
        raise ValueError(f"Unknown search engine name '{search_engine_name}' in "
                         f"SEARCH_ENGINE_NAME environment variable.")
    
    def search_with_tavily(
        self, keywords: str, max_results: Optional[int] = 10, 
        region: Optional[str] = None
    ) -> list:
        """
        Drop-in replacement for BFCL web search tool, using a wrapper around the Tavily
        search API that makes its results look similar to the BFCL web search tool. 
        

        See search_engine_query_original() for full docs.
        """
        country = _duckduckgo_region_code_to_tavily_country_code(region)
        kwargs = {
            "query": keywords,
            "search_depth": "basic",
            "max_results": max_results if max_results else 10,
        }
        if country is not None:
            kwargs["country"] = country

        tavily_key = os.getenv("TAVILY_API_KEY")
        #print(f"Tavily API key: {tavily_key}")
        if tavily_key is None:
            raise ValueError("Required environment variable TAVILY_API_KEY not set.")

        tavily_client = tavily.TavilyClient(tavily_key)

        # Replicate exponential backoff behavior of original tool
        backoff = 2  # initial back-off in seconds
        num_attempts_remaining = 10  # Just in case
        while num_attempts_remaining > 0:
            num_attempts_remaining -= 1
            try:
                response = tavily_client.search(**kwargs)
                break
            except tavily.exceptions.UsageLimitExceededError:
                wait_time = backoff + random.uniform(0, backoff)
                error_block = (
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Received 429 from Tavily search. "
                    f"Retrying in {wait_time:.1f} seconds…\n"
                    + "*" * 100
                )
                print(error_block)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)  # cap the back-off
                continue
            
        if num_attempts_remaining == 0:
            raise ValueError("Failed to reach Tavily after 10 attempts.")

        return [
            {
                "title": r["title"],
                "href": r["url"],
                # Tavily's snippets are much longer than SerpAPI's. We shorten them to 
                # produce similar results when running the benchmark.
                "body": shorten_snippet(
                    keywords, r["content"], max_chars=300, hard_max_chars=500
                ),
            }
            for r in response["results"]
        ]

    def search_with_ibm_mcp(
        self, keywords: str, max_results: Optional[int] = 10
    ) -> list:
        """
        Drop-in replacement for BFCL web search tool, using our internal MCP search tool
        for code agents. No region support in the internal tool.

        See search_engine_query_original() for full docs.
        """
        if "MCP_SEARCH_TOOL_URL" not in os.environ:
            raise ValueError(
                "Please set the MCP_SEARCH_TOOL_URL environment variable to the URL "
                "of the MCP web search service."
            )
        mcp_base_url = os.environ["MCP_SEARCH_TOOL_URL"]
        
        backoff = 20  # initial back-off in seconds
        max_retries = 10

        with httpx.Client(verify=True) as client:
            # Protocol intitialization ritual, see
            # https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle

            # Step 1: Initialize
            init_response = client.post(
                mcp_base_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,  # Must be different for each request
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "python-mcp-client", "version": "1.0.0"},
                    },
                },
            )
            init_response.raise_for_status()
            # init_response_json = json.loads(init_response.content)

            # Initialization returns a session id in an HTTP header, not the response
            session_id = init_response.headers["mcp-session-id"] or None

            # Step 2: Unnecessary acknowledgement of initialization message
            init_some_more_response = client.post(
                mcp_base_url,
                json={
                    "jsonrpc": "2.0",
                    # No ID for some reason, per the protocol spec
                    "method": "notifications/initialized",
                },
                headers={"mcp-session-id": session_id},
            )
            init_some_more_response.raise_for_status()

            # Step 3: Make a tool call
            num_retries = 0
            while num_retries < max_retries:
                tool_call_response = client.post(
                    mcp_base_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "google_pse_search",
                            "arguments": {"query": keywords, "num_results": max_results},
                        },
                    },
                    headers={"mcp-session-id": session_id},
                )
                tool_call_response.raise_for_status()
                tool_call_response_json = tool_call_response.json()
                
                # Search tool stuffs structured data into "human-readable" text, 
                # wrapped in JSON.
                # Extract the text.
                search_result_text = tool_call_response_json["result"]["content"][0][
                    "text"
                ]
                
                # Sometimes the human-readable text contains a JSON error message.
                # Use the same backoff code as the original implementation, but with
                # bounded retries
                if "google API error (status 429)" in search_result_text:
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100
                        + "\n❗️❗️ [WebSearchAPI] Received 429 from search API. "
                        + f"Retrying in {wait_time:.1f} seconds…\n"
                        + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                break  # Success – no rate-limit error detected

        try:
            # Parse the results of the tool call into the format of the original BFCL 
            # tool

            # First line is search query and number of results
            query_part, result_part = search_result_text.split("\n\n", maxsplit=1)

            # Parse first line with a regex.
            # Do this defensively because we have no idea what's on the other side of this MCP call
            matcher = re.match(r"Found (\d+) results for: (.+)", query_part)
            if not matcher:
                raise ValueError(f"Couldn't parse first line '{query_part}'")
            num_results_returned, query_returned = int(matcher.group(1)), matcher.group(2)
            if num_results_returned > max_results:
                raise ValueError(
                    f"Requested {max_results} results but received {num_results_returned}"
                )
            if keywords != query_returned:
                raise ValueError(
                    f"Tried to run query '{keywords}' but ran query '{query_returned}' instead."
                )

            result_strs = result_part.split("\n\n")
            # Extra \n\n at end
            result_strs = result_strs[:-1]

            results = []
            for s in result_strs:
                title_part, url_part, summary_part = s.split("\n   ")

                # Title part contains a number.
                matcher = re.match(r"\d+\. (.+)", title_part)
                title = matcher.group(1)

                result = {
                    "title": title,
                    # Second line is just URL with a hanging indent we've already removed
                    "href": url_part,
                }
                if self.show_snippet:
                    # Third line is summary with additional cruft for optional timestamp,
                    # e.g."Feb 6, 2025 ... snippet snippet snippet"
                    # Leave the additional cruft in place for now.
                    result["body"] = summary_part
                results.append(result)

            return results
        except ValueError as e:
            raise ValueError(
                f"Error parsing search result. Result was:\n{search_result_text}"
            ) from e

    def search_engine_query_original(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.
        """
        backoff = 2  # initial back-off in seconds
        params = {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }

        # Infinite retry loop with exponential backoff
        while True:
            try:
                search = GoogleSearch(params)
                search_results = search.get_dict()
            except Exception as e:
                # If the underlying HTTP call raised a 429 we retry, otherwise propagate
                if "429" in str(e):
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                        + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)  # cap the back-off
                    continue
                else:
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Error from SerpAPI: {str(e)}. This is not a rate-limit error, so it will not be retried."
                        + "*" * 100
                    )
                    print(error_block)
                    return {"error": str(e)}

            # SerpAPI sometimes returns the error in the payload instead of raising
            if "error" in search_results and "429" in str(search_results["error"]):
                wait_time = backoff + random.uniform(0, backoff)
                error_block = (
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                    + "*" * 100
                )
                print(error_block)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                continue

            break  # Success – no rate-limit error detected

        if "organic_results" not in search_results:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        search_results = search_results["organic_results"]

        # Convert the search results to the desired format
        results = []
        for result in search_results[:max_results]:
            if self.show_snippet:
                results.append(
                    {
                        "title": result["title"],
                        "href": result["link"],
                        "body": result["snippet"],
                    }
                )
            else:
                results.append(
                    {
                        "title": result["title"],
                        "href": result["link"],
                    }
                )

        return results

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            # A header that mimics a browser request. This helps avoid 403 Forbidden errors.
            # TODO: Is this the best way to do this?
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/112.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://www.google.com/",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
            }
            response = requests.get(
                url, headers=headers, timeout=20, allow_redirects=True
            )
            response.raise_for_status()

            # Note: Un-comment this when we want to simulate a random error
            # Flip a coin to simulate a random error
            # if self._random.random() < 0.95:
            #     return {"error": self._fake_requests_get_error_msg(url)}

            # Process the response based on the mode
            if mode == "raw":
                return {"content": response.text}

            elif mode == "markdown":
                converter = html2text.HTML2Text()
                markdown = converter.handle(response.text)
                return {"content": markdown}

            elif mode == "truncate":
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove scripts and styles
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                # Extract and clean text
                text = soup.get_text(separator="\n", strip=True)
                return {"content": text}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            return {"error": f"An error occurred while fetching {url}: {str(e)}"}

    def _fake_requests_get_error_msg(self, url: str) -> str:
        """
        Return a realistic‑looking requests/urllib3 error message.
        """
        parsed = urlparse(url)

        context = {
            "url": url,
            "host": parsed.hostname or "unknown",
            "path": parsed.path or "/",
            "id1": self._rng.randrange(0x10000000, 0xFFFFFFFF),
            "id2": self._rng.randrange(0x10000000, 0xFFFFFFFF),
        }

        template = self._rng.choice(ERROR_TEMPLATES)

        return template.format(**context)
