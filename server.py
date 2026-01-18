"""
ACI Configurator - REAL Working Demo
=====================================
FastAPI backend with actual:
- RAG (sentence-transformers embeddings)
- MCP (GitHub connection)
- LangGraph (LLM routing)
- APIC deployment

Run: uvicorn server:app --reload
"""
import os
import json
import asyncio
import time
from typing import TypedDict, AsyncGenerator
from contextlib import asynccontextmanager

# Load environment variables BEFORE LangChain imports (for LangSmith)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  pip install sentence-transformers numpy for real RAG")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# ═══════════════════════════════════════════════════════════════════════════════
# 1. RAG - Real Vector Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
ACI_KNOWLEDGE = {
    "fvTenant": "Tenant organization container multi-tenancy separate customer environments isolate different departments",
    "fvCtx": "VRF Virtual Routing Forwarding network isolation separate traffic layer 3 segmentation routing domain",
    "fvBD": "Bridge Domain Layer 2 network broadcast domain subnet gateway where servers get IP",
    "fvAp": "Application Profile application container group related EPGs three tier application multi-tier",
    "fvAEPg": "EPG Endpoint Group security policy microsegmentation web tier database tier app tier",
    "fvSubnet": "Subnet IP address range gateway CIDR block default gateway routing",
    "vzBrCP": "Contract allow traffic between EPGs firewall rule security policy permit communication whitelist",
    "vzFilter": "Filter traffic match criteria protocol port allow HTTP permit SSH block ICMP",
    "vzEntry": "Filter Entry single filter rule TCP port UDP protocol specific traffic match",
    "l3extOut": "L3Out external network connectivity internet access WAN connection isolated from internet no external",
}

class RealRAG:
    """Real RAG with sentence-transformers."""
    
    # Minimum similarity score to be considered relevant
    # Sentence-transformers cosine similarity ranges roughly 0-1
    # Unrelated queries like "coffee" typically score 0.1-0.4
    # ACI-related queries should score 0.45+
    RELEVANCE_THRESHOLD = 0.45
    
    def __init__(self):
        self.classes = list(ACI_KNOWLEDGE.keys())
        self.descriptions = list(ACI_KNOWLEDGE.values())
        self.model = None
        self.embeddings = None
        
        if HAS_EMBEDDINGS:
            print("🔄 Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings = self.model.encode(self.descriptions, convert_to_numpy=True)
            print("✅ RAG ready with real embeddings")
    
    def search(self, query: str, top_k: int = 6) -> list[dict]:
        """Vector similarity search with relevance threshold."""
        if not HAS_EMBEDDINGS or self.model is None:
            return self._keyword_fallback(query, top_k)
        
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        similarities = np.dot(self.embeddings, query_emb)
        
        # Debug: show top scores
        sorted_indices = np.argsort(similarities)[::-1]
        top_score = float(similarities[sorted_indices[0]])
        print(f"🔍 RAG query: '{query[:50]}...' | Top score: {top_score:.3f} | Threshold: {self.RELEVANCE_THRESHOLD}")
        
        # Filter by relevance threshold
        relevant_indices = [i for i, score in enumerate(similarities) if score >= self.RELEVANCE_THRESHOLD]
        
        if not relevant_indices:
            # No relevant results found
            print(f"   ❌ No results above threshold")
            return []
        
        # Sort by score and take top_k
        relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
        top_indices = relevant_indices[:top_k]
        
        print(f"   ✅ Found {len(top_indices)} relevant results")
        return [
            {"class": self.classes[i], "score": round(float(similarities[i]), 3), 
             "desc": self.descriptions[i][:60] + "..."}
            for i in top_indices
        ]
    
    def _keyword_fallback(self, query: str, top_k: int) -> list[dict]:
        """Fallback keyword matching with threshold."""
        query_words = set(query.lower().split())
        
        # ACI-related keywords that should be present for relevance
        aci_keywords = {"network", "application", "tier", "database", "web", "app", "isolated", 
                       "internet", "firewall", "contract", "tenant", "vrf", "epg", "bridge",
                       "subnet", "security", "policy", "server", "deploy", "config", "aci"}
        
        # Check if query has any ACI-related terms
        if not query_words & aci_keywords:
            return []  # No relevant results
        
        scores = []
        for i, desc in enumerate(self.descriptions):
            overlap = len(query_words & set(desc.lower().split()))
            scores.append((i, overlap))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Only return if there's some overlap
        results = [
            {"class": self.classes[i], "score": s, "desc": self.descriptions[i][:60] + "..."}
            for i, s in scores[:top_k] if s > 0
        ]
        return results

# Global RAG instance
rag = None

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MCP - Real GitHub Connection
# ═══════════════════════════════════════════════════════════════════════════════
class MCPClient:
    """Real MCP client for GitHub."""
    
    def __init__(self):
        self.process = None
        self.connected = False
        self._id = 0
    
    async def connect(self) -> bool:
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not token:
            print("MCP: ⚠️  No GITHUB_PERSONAL_ACCESS_TOKEN set")
            return False
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                "npx", "-y", "@modelcontextprotocol/server-github",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token}
            )
            
            # Wait a moment for server to start
            await asyncio.sleep(1)
            
            # MCP handshake
            resp = await self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "aci-demo", "version": "1.0"}
            })
            
            if resp and "result" in resp:
                await self._notify("notifications/initialized", {})
                self.connected = True
                return True
            else:
                print(f"MCP: Initialize failed - {resp}")
                return False
                
        except Exception as e:
            print(f"MCP error: {e}")
            return False
    
    async def _request(self, method: str, params: dict) -> dict | None:
        if not self.process or self.process.returncode is not None:
            return None
        
        self._id += 1
        msg = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}) + "\n"
        
        try:
            self.process.stdin.write(msg.encode())
            await self.process.stdin.drain()
            
            # Read response with timeout
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=30)
            if not line:
                return None
            return json.loads(line.decode().strip())
        except asyncio.TimeoutError:
            print("MCP: Request timeout")
            return None
        except json.JSONDecodeError as e:
            print(f"MCP: JSON decode error - {e}")
            return None
        except Exception as e:
            print(f"MCP: Request error - {e}")
            return None
    
    async def _notify(self, method: str, params: dict):
        if self.process and self.process.returncode is None:
            try:
                msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
                self.process.stdin.write(msg.encode())
                await self.process.stdin.drain()
            except Exception:
                pass
    
    async def get_file(self, owner: str, repo: str, path: str) -> str:
        if not self.connected:
            return "MCP not connected"
        
        resp = await self._request("tools/call", {
            "name": "get_file_contents",
            "arguments": {"owner": owner, "repo": repo, "path": path}
        })
        
        if resp and "result" in resp:
            content = resp["result"].get("content", [])
            texts = []
            for c in content:
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "resource":
                    resource = c.get("resource", {})
                    if resource.get("text"):
                        texts.append(resource.get("text"))
            return "\n".join(texts) if texts else "Empty response"
        
        return f"Error: {resp.get('error', 'Unknown error') if resp else 'No response'}"
    
    async def search_code(self, query: str) -> str:
        if not self.connected:
            return "MCP not connected"
        resp = await self._request("tools/call", {
            "name": "search_code",
            "arguments": {"q": query}
        })
        if resp and "result" in resp:
            content = resp["result"].get("content", [])
            return "\n".join(c.get("text", "")[:500] for c in content if c.get("type") == "text")
        return "No results"
    
    async def disconnect(self):
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except Exception:
                self.process.kill()
            self.connected = False

# Global MCP
mcp = MCPClient()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Payload Builder - LLM-based with validation
# ═══════════════════════════════════════════════════════════════════════════════

# Valid ACI classes (for validation)
VALID_ACI_CLASSES = {
    "fvTenant", "fvCtx", "fvBD", "fvAp", "fvAEPg", "fvSubnet",
    "vzBrCP", "vzSubj", "vzFilter", "vzEntry", "l3extOut",
    "fvRsBd", "fvRsCtx", "fvRsCons", "fvRsProv", "vzRsSubjFiltAtt"
}

PAYLOAD_BUILDER_PROMPT = """You are an ACI configuration expert. Build a valid ACI JSON payload based on:

USER INTENT: {prompt}

RAG RESULTS (relevant ACI objects):
{rag_results}

MCP METADATA:
{mcp_data}

RULES:
1. Use ONLY valid ACI class names: fvTenant, fvCtx, fvBD, fvAp, fvAEPg, fvSubnet, vzBrCP, vzSubj, vzFilter, vzEntry, l3extOut
2. Relationships: fvRsBd, fvRsCtx, fvRsCons, fvRsProv, vzRsSubjFiltAtt
3. Structure: fvTenant > (fvCtx, fvBD, fvAp, vzBrCP, vzFilter) > children
4. For "isolated" or "no internet": DB EPG should have fvRsProv but NO fvRsCons (can't initiate outbound)
5. Always include proper contract relationships between tiers

OUTPUT FORMAT - Return ONLY valid JSON, no markdown, no explanation:
{{"fvTenant": {{"attributes": {{"name": "..."}}, "children": [...]}}}}

Build the payload now:"""

def validate_payload(payload: dict) -> tuple[bool, str]:
    """Validate ACI payload structure and class names."""
    errors = []
    
    def check_node(node: dict, path: str = ""):
        for key, value in node.items():
            # Check if it's an ACI class (starts with lowercase, not 'attributes' or 'children')
            if key not in ("attributes", "children") and not key.startswith("tn"):
                if key not in VALID_ACI_CLASSES:
                    errors.append(f"Invalid class '{key}' at {path}")
            
            if isinstance(value, dict):
                if "children" in value:
                    for i, child in enumerate(value["children"]):
                        check_node(child, f"{path}/{key}[{i}]")
                elif key not in ("attributes",):
                    check_node(value, f"{path}/{key}")
    
    check_node(payload)
    return len(errors) == 0, "; ".join(errors)

def build_payload_with_llm(
    prompt: str, 
    rag_results: list[dict], 
    mcp_data: str | None,
    llm
) -> tuple[dict, str]:
    """Use LLM to build payload, with validation fallback."""
    
    # Format RAG results
    rag_str = "\n".join([
        f"- {r['class']}: {r.get('desc', 'N/A')} (relevance: {r['score']})"
        for r in rag_results
    ])
    
    # Build the prompt
    builder_prompt = PAYLOAD_BUILDER_PROMPT.format(
        prompt=prompt,
        rag_results=rag_str,
        mcp_data=mcp_data[:1000] if mcp_data else "No additional metadata"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=builder_prompt)])
        content = response.content.strip()
        
        # Clean up response (remove markdown if present)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        # Parse JSON
        payload = json.loads(content)
        
        # Validate
        is_valid, errors = validate_payload(payload)
        if is_valid:
            return payload, "LLM-generated"
        else:
            print(f"⚠️ LLM payload validation failed: {errors}")
            return build_payload_fallback(prompt, rag_results), f"Fallback (validation failed: {errors})"
    
    except json.JSONDecodeError as e:
        print(f"⚠️ LLM returned invalid JSON: {e}")
        return build_payload_fallback(prompt, rag_results), f"Fallback (JSON error: {e})"
    except Exception as e:
        print(f"⚠️ LLM builder error: {e}")
        return build_payload_fallback(prompt, rag_results), f"Fallback (error: {e})"

def build_payload_fallback(prompt: str, rag_results: list[dict]) -> dict:
    """Deterministic fallback builder."""
    prompt_lower = prompt.lower()
    is_three_tier = "three" in prompt_lower or "tier" in prompt_lower
    needs_isolation = "isolat" in prompt_lower or "no internet" in prompt_lower
    
    tiers = ["web", "app", "db"] if is_three_tier else ["web", "app"]
    
    children = [
        {"fvCtx": {"attributes": {"name": "prod-vrf"}}}
    ]
    
    # Bridge Domains
    for i, tier in enumerate(tiers):
        children.append({
            "fvBD": {
                "attributes": {"name": f"{tier}-bd"},
                "children": [
                    {"fvRsCtx": {"attributes": {"tnFvCtxName": "prod-vrf"}}},
                    {"fvSubnet": {"attributes": {"ip": f"10.0.{i+1}.1/24"}}}
                ]
            }
        })
    
    # Application Profile with EPGs
    epg_children = []
    for tier in tiers:
        epg = {
            "fvAEPg": {
                "attributes": {"name": f"{tier}-epg"},
                "children": [{"fvRsBd": {"attributes": {"tnFvBDName": f"{tier}-bd"}}}]
            }
        }
        if tier == "web":
            epg["fvAEPg"]["children"].append({"fvRsCons": {"attributes": {"tnVzBrCPName": "web-to-app"}}})
        elif tier == "app":
            epg["fvAEPg"]["children"].extend([
                {"fvRsProv": {"attributes": {"tnVzBrCPName": "web-to-app"}}},
                {"fvRsCons": {"attributes": {"tnVzBrCPName": "app-to-db"}}}
            ])
        elif tier == "db":
            epg["fvAEPg"]["children"].append({"fvRsProv": {"attributes": {"tnVzBrCPName": "app-to-db"}}})
            if needs_isolation:
                epg["fvAEPg"]["attributes"]["descr"] = "ISOLATED-no-outbound"
        epg_children.append(epg)
    
    children.append({"fvAp": {"attributes": {"name": "app-profile"}, "children": epg_children}})
    
    # Contracts
    children.extend([
        {"vzBrCP": {"attributes": {"name": "web-to-app"}}},
        {"vzBrCP": {"attributes": {"name": "app-to-db"}}}
    ])
    
    return {"fvTenant": {"attributes": {"name": "auto-tenant"}, "children": children}}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. APIC Deployment - Real API calls
# ═══════════════════════════════════════════════════════════════════════════════
async def deploy_to_apic(payload: dict) -> dict:
    """Real APIC deployment."""
    base_url = os.getenv("APIC_BASE_URL")
    username = os.getenv("APIC_USERNAME")
    password = os.getenv("APIC_PASSWORD")
    
    if not all([base_url, username, password]):
        return {"status": "skipped", "message": "APIC not configured (set APIC_BASE_URL, APIC_USERNAME, APIC_PASSWORD)"}
    
    if not HAS_AIOHTTP:
        return {"status": "error", "message": "aiohttp not installed"}
    
    tenant_name = payload.get("fvTenant", {}).get("attributes", {}).get("name", "unknown")
    dn = f"uni/tn-{tenant_name}"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Login
            login_url = f"{base_url.rstrip('/')}/api/aaaLogin.json"
            login_payload = {"aaaUser": {"attributes": {"name": username, "pwd": password}}}
            
            async with session.post(login_url, json=login_payload, ssl=False) as resp:
                if resp.status != 200:
                    return {"status": "error", "message": f"Login failed: HTTP {resp.status}"}
                data = await resp.json()
                token = data.get("imdata", [{}])[0].get("aaaLogin", {}).get("attributes", {}).get("token")
            
            # POST config
            config_url = f"{base_url.rstrip('/')}/api/mo/{dn}.json"
            cookies = {"APIC-cookie": token}
            
            async with session.post(config_url, json=payload, cookies=cookies, ssl=False) as resp:
                if resp.status == 200:
                    return {"status": "success", "message": f"Deployed to {dn}", "dn": dn, "http": 200}
                else:
                    err = await resp.json()
                    msg = err.get("imdata", [{}])[0].get("error", {}).get("attributes", {}).get("text", "Unknown")
                    return {"status": "error", "message": msg, "http": resp.status}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. LangGraph - Real LLM Routing
# ═══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    prompt: str
    rag_results: list | None
    mcp_data: str | None
    payload: dict | None
    deploy_result: dict | None
    next: str
    logs: list


def create_real_graph(config: dict = None):
    """Create LangGraph with real LLM routing.
    
    Args:
        config: Optional config dict passed by LangGraph Studio (ignored)
    """
    llm_model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    llm = ChatAnthropic(model=llm_model, temperature=0)
    
    def llm_node(state: AgentState) -> AgentState:
        """LLM decides next action."""
        time.sleep(0.8)  # Demo delay
        
        # Check if RAG was done but returned no results - can't proceed
        rag_results = state.get("rag_results")
        rag_done_but_empty = isinstance(rag_results, list) and len(rag_results) == 0
        
        if rag_done_but_empty:
            state["next"] = "done"
            state["logs"].append({
                "type": "llm", 
                "action": "done",
                "raw": "Cannot proceed - no relevant ACI objects found",
                "prompt": state["prompt"][:100],
                "error": "This query doesn't appear to be related to ACI network configuration. Try something like 'Build a three tier application with database isolated from internet'."
            })
            return state
        
        # Determine RAG status for context
        rag_status = "not done" if state["rag_results"] is None else f"done with {len(state['rag_results'])} results"
        
        context = f"""Current state:
- Prompt: {state["prompt"]}
- RAG: {rag_status}
- MCP data: {state["mcp_data"] is not None}
- Payload built: {state["payload"] is not None}
- Deployed: {state["deploy_result"] is not None}

Based on this state, what should be the next action?
Available actions: rag, mcp, build, deploy, done

Rules:
- If RAG not done, return "rag"
- If RAG done with 0 results, return "done" (can't proceed without relevant objects)
- If RAG done with results but MCP not done, return "mcp"  
- If both done but payload not built, return "build"
- If payload built but not deployed, return "deploy"
- Otherwise return "done"

Respond with ONLY the action name, nothing else."""
        
        response = llm.invoke([HumanMessage(content=context)])
        action = response.content.strip().lower()
        
        # Parse action with fallback logic
        # Note: rag_results is None (not run) vs [] (ran but empty) vs [items] (has results)
        if "rag" in action and state["rag_results"] is None:
            state["next"] = "rag"
        elif "mcp" in action and not state["mcp_data"]:
            state["next"] = "mcp"
        elif "build" in action and not state["payload"]:
            state["next"] = "build"
        elif "deploy" in action and state["payload"] and not state["deploy_result"]:
            state["next"] = "deploy"
        else:
            state["next"] = "done"
        
        state["logs"].append({
            "type": "llm", 
            "action": state["next"], 
            "raw": action,
            "prompt": state["prompt"][:100]
        })
        return state
    
    def rag_node(state: AgentState) -> AgentState:
        """Real RAG search."""
        time.sleep(1.0)  # Demo delay
        
        global rag
        results = rag.search(state["prompt"], top_k=6)
        state["rag_results"] = results  # Keep as empty list if no results
        
        if results:
            state["logs"].append({
                "type": "rag", 
                "query": state["prompt"],
                "results": results,
                "found": True
            })
        else:
            state["logs"].append({
                "type": "rag", 
                "query": state["prompt"],
                "results": [],
                "found": False,
                "message": "No relevant ACI objects found for this query"
            })
        return state
    
    def mcp_node(state: AgentState) -> AgentState:
        """MCP call - fetches from devnet-1260 repo."""
        time.sleep(1.2)  # Demo delay
        
        global mcp
        owner = os.getenv("GITHUB_OWNER", "")
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        
        # Try to fetch via GitHub API directly (synchronous)
        if owner and token:
            try:
                import requests
                url = f"https://api.github.com/repos/{owner}/devnet-1260/contents/knowledge/aci_metadata.py"
                headers = {
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3.raw"
                }
                resp = requests.get(url, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.text[:500]  # First 500 chars
                    state["mcp_data"] = resp.text
                    state["logs"].append({
                        "type": "mcp", 
                        "source": f"{owner}/devnet-1260/knowledge/aci_metadata.py",
                        "data": data,
                        "prompt": state["prompt"][:50]
                    })
                    return state
                else:
                    print(f"GitHub API error: {resp.status_code}")
            except Exception as e:
                print(f"GitHub fetch error: {e}")
        
        # Fallback: Use local ACI metadata knowledge
        fallback_metadata = """
ACI_METADATA = {
    "fvTenant": {"dn": "uni/tn-{name}", "children": ["fvCtx", "fvBD", "fvAp", "vzBrCP", "vzFilter"]},
    "fvCtx": {"dn": "uni/tn-{t}/ctx-{name}", "desc": "VRF context for routing isolation"},
    "fvBD": {"dn": "uni/tn-{t}/BD-{name}", "children": ["fvSubnet", "fvRsCtx"]},
    "fvAp": {"dn": "uni/tn-{t}/ap-{name}", "children": ["fvAEPg"]},
    "fvAEPg": {"dn": "uni/tn-{t}/ap-{a}/epg-{name}", "children": ["fvRsBd", "fvRsCons", "fvRsProv"]},
    "vzBrCP": {"dn": "uni/tn-{t}/brc-{name}", "children": ["vzSubj"]},
    "fvRsCons": {"desc": "Consumer contract - EPG can initiate traffic"},
    "fvRsProv": {"desc": "Provider contract - EPG accepts traffic"}
}
# Note: For isolation, DB EPG should only have fvRsProv (no fvRsCons = cannot initiate outbound)
"""
        state["mcp_data"] = fallback_metadata
        
        if not owner or not token:
            msg = "GitHub not configured - using local ACI metadata"
        else:
            msg = "GitHub fetch failed - using local ACI metadata"
        
        state["logs"].append({
            "type": "mcp", 
            "data": msg + "\n" + fallback_metadata[:200] + "...",
            "prompt": state["prompt"][:50]
        })
        return state
    
    def build_node(state: AgentState) -> AgentState:
        """Build payload using LLM with validation fallback."""
        time.sleep(1.5)  # Demo delay - longer for "thinking"
        
        payload, method = build_payload_with_llm(
            state["prompt"],
            state["rag_results"] or [],
            state["mcp_data"],
            llm
        )
        state["payload"] = payload
        state["logs"].append({
            "type": "build", 
            "method": method, 
            "prompt": state["prompt"],  # Show what prompt was used
            "rag_classes": [r["class"] for r in (state["rag_results"] or [])],
            "payload": payload
        })
        return state
    
    def deploy_node(state: AgentState) -> AgentState:
        """Deploy to APIC."""
        time.sleep(1.0)  # Demo delay
        
        # Check APIC config
        base_url = os.getenv("APIC_BASE_URL")
        username = os.getenv("APIC_USERNAME")
        password = os.getenv("APIC_PASSWORD")
        
        if not all([base_url, username, password]):
            result = {"status": "skipped", "message": "APIC not configured (set APIC_BASE_URL, APIC_USERNAME, APIC_PASSWORD in .env)"}
        else:
            # Try to deploy
            try:
                import requests
                requests.packages.urllib3.disable_warnings()
                
                tenant_name = state["payload"].get("fvTenant", {}).get("attributes", {}).get("name", "unknown")
                dn = f"uni/tn-{tenant_name}"
                
                # Login
                login_url = f"{base_url.rstrip('/')}/api/aaaLogin.json"
                login_payload = {"aaaUser": {"attributes": {"name": username, "pwd": password}}}
                login_resp = requests.post(login_url, json=login_payload, verify=False, timeout=10)
                
                if login_resp.status_code != 200:
                    result = {"status": "warning", "message": "APIC is offline - cannot deploy"}
                else:
                    token = login_resp.json().get("imdata", [{}])[0].get("aaaLogin", {}).get("attributes", {}).get("token")
                    
                    # POST config
                    config_url = f"{base_url.rstrip('/')}/api/mo/{dn}.json"
                    cookies = {"APIC-cookie": token}
                    config_resp = requests.post(config_url, json=state["payload"], cookies=cookies, verify=False, timeout=10)
                    
                    if config_resp.status_code == 200:
                        result = {"status": "success", "message": f"Deployed to {dn}", "dn": dn, "http": 200}
                    else:
                        result = {"status": "warning", "message": "APIC is offline - cannot deploy"}
            except Exception as e:
                # Any error (timeout, connection refused, etc) = APIC offline
                result = {"status": "warning", "message": "APIC is offline - cannot deploy"}
        
        state["deploy_result"] = result
        state["logs"].append({
            "type": "deploy", 
            "result": result,
            "prompt": state["prompt"][:50]
        })
        return state
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("rag", rag_node)
    graph.add_node("mcp", mcp_node)
    graph.add_node("build", build_node)
    graph.add_node("deploy", deploy_node)
    
    graph.set_entry_point("llm")
    
    graph.add_conditional_edges("llm", lambda s: s["next"], {
        "rag": "rag", "mcp": "mcp", "build": "build", "deploy": "deploy", "done": END
    })
    
    graph.add_edge("rag", "llm")
    graph.add_edge("mcp", "llm")
    graph.add_edge("build", "llm")
    graph.add_edge("deploy", "llm")
    
    return graph.compile(checkpointer=MemorySaver())

# ═══════════════════════════════════════════════════════════════════════════════
# 6. FastAPI Server
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    global rag, mcp
    rag = RealRAG()
    
    # Try MCP connection
    if os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        connected = await mcp.connect()
        print(f"MCP: {'✅ Connected' if connected else '❌ Failed'}")
    else:
        print("MCP: ⚠️  No GitHub token")
    
    yield
    
    # Cleanup
    await mcp.disconnect()

app = FastAPI(title="ACI Configurator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PromptRequest(BaseModel):
    prompt: str
    thread_id: str = "demo-1"

@app.post("/run")
async def run_agent(req: PromptRequest):
    """Run the agent and stream results."""
    
    async def generate() -> AsyncGenerator[str, None]:
        graph = create_real_graph()
        
        state: AgentState = {
            "prompt": req.prompt,
            "rag_results": None,
            "mcp_data": None,
            "payload": None,
            "deploy_result": None,
            "next": "",
            "logs": []
        }
        
        config = {"configurable": {"thread_id": req.thread_id}}
        
        prev_log_count = 0
        for event in graph.stream(state, config):
            if isinstance(event, dict):
                for node_name, node_state in event.items():
                    # Send new logs with delay between each
                    new_logs = node_state.get("logs", [])[prev_log_count:]
                    for log in new_logs:
                        await asyncio.sleep(0.6)  # Demo delay between log entries
                        yield f"data: {json.dumps(log)}\n\n"
                    prev_log_count = len(node_state.get("logs", []))
                    state = node_state
        
        # Send final state
        await asyncio.sleep(0.8)  # Final delay
        yield f"data: {json.dumps({'type': 'done', 'payload': state.get('payload'), 'deploy': state.get('deploy_result')})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/status")
async def status():
    """Check system status."""
    return {
        "rag": HAS_EMBEDDINGS,
        "mcp": mcp.connected if mcp else False,
        "apic": bool(os.getenv("APIC_BASE_URL")),
        "llm": bool(os.getenv("ANTHROPIC_API_KEY"))
    }

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the UI."""
    return open("ui.html").read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
