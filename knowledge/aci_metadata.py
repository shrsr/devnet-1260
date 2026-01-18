"""
Minimal ACI Metadata for DEVNET-1260 Demo
=========================================
MCP reads this file from GitHub to get object schemas and relationships.
"""

ACI_METADATA = {
    "fvTenant": {
        "dn_format": "uni/tn-{name}",
        "required": ["name"],
        "children": ["fvCtx", "fvBD", "fvAp", "vzBrCP", "vzFilter"],
        "description": "Tenant - top level container for network policies"
    },
    "fvCtx": {
        "dn_format": "uni/tn-{tenant}/ctx-{name}",
        "required": ["name"],
        "parent": "fvTenant",
        "description": "VRF - Layer 3 isolation domain"
    },
    "fvBD": {
        "dn_format": "uni/tn-{tenant}/BD-{name}",
        "required": ["name"],
        "children": ["fvSubnet", "fvRsCtx"],
        "parent": "fvTenant",
        "description": "Bridge Domain - Layer 2 broadcast domain"
    },
    "fvSubnet": {
        "dn_format": "uni/tn-{tenant}/BD-{bd}/subnet-[{ip}]",
        "required": ["ip"],
        "parent": "fvBD",
        "description": "Subnet - IP gateway for bridge domain"
    },
    "fvAp": {
        "dn_format": "uni/tn-{tenant}/ap-{name}",
        "required": ["name"],
        "children": ["fvAEPg"],
        "parent": "fvTenant",
        "description": "Application Profile - groups related EPGs"
    },
    "fvAEPg": {
        "dn_format": "uni/tn-{tenant}/ap-{ap}/epg-{name}",
        "required": ["name"],
        "children": ["fvRsBd", "fvRsCons", "fvRsProv"],
        "parent": "fvAp",
        "description": "Endpoint Group - security zone for endpoints"
    },
    "vzBrCP": {
        "dn_format": "uni/tn-{tenant}/brc-{name}",
        "required": ["name"],
        "children": ["vzSubj"],
        "parent": "fvTenant",
        "description": "Contract - defines allowed traffic between EPGs"
    },
    "vzSubj": {
        "dn_format": "uni/tn-{tenant}/brc-{contract}/subj-{name}",
        "required": ["name"],
        "children": ["vzRsSubjFiltAtt"],
        "parent": "vzBrCP",
        "description": "Contract Subject - groups filters in a contract"
    },
    "vzFilter": {
        "dn_format": "uni/tn-{tenant}/flt-{name}",
        "required": ["name"],
        "children": ["vzEntry"],
        "parent": "fvTenant",
        "description": "Filter - Layer 4 traffic rules"
    },
    "vzEntry": {
        "dn_format": "uni/tn-{tenant}/flt-{filter}/e-{name}",
        "required": ["name", "etherT"],
        "parent": "vzFilter",
        "properties": {
            "etherT": ["ip", "arp", "unspecified"],
            "prot": ["tcp", "udp", "icmp", "unspecified"],
            "dFromPort": "destination port start",
            "dToPort": "destination port end"
        },
        "description": "Filter Entry - single L4 rule (protocol/port)"
    },
    
    # Relationship objects
    "fvRsCtx": {
        "dn_format": "uni/tn-{tenant}/BD-{bd}/rsctx",
        "required": ["tnFvCtxName"],
        "parent": "fvBD",
        "description": "Links Bridge Domain to VRF"
    },
    "fvRsBd": {
        "dn_format": "uni/tn-{tenant}/ap-{ap}/epg-{epg}/rsbd",
        "required": ["tnFvBDName"],
        "parent": "fvAEPg",
        "description": "Links EPG to Bridge Domain"
    },
    "fvRsCons": {
        "dn_format": "uni/tn-{tenant}/ap-{ap}/epg-{epg}/rscons-{contract}",
        "required": ["tnVzBrCPName"],
        "parent": "fvAEPg",
        "description": "EPG consumes a contract (client side)"
    },
    "fvRsProv": {
        "dn_format": "uni/tn-{tenant}/ap-{ap}/epg-{epg}/rsprov-{contract}",
        "required": ["tnVzBrCPName"],
        "parent": "fvAEPg",
        "description": "EPG provides a contract (server side)"
    },
    "vzRsSubjFiltAtt": {
        "dn_format": "uni/tn-{tenant}/brc-{contract}/subj-{subj}/rssubjFiltAtt-{filter}",
        "required": ["tnVzFilterName"],
        "parent": "vzSubj",
        "description": "Links contract subject to filter"
    }
}
