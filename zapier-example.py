import requests

# Your Zapier MCP endpoint
ZAPIER_MCP_URL = 'https://mcp.zapier.com/your_unique_id'

def execute_mcp_action(prompt):
    response = requests.post(
        ZAPIER_MCP_URL,
        json={'prompt': prompt},
        headers={'Content-Type': 'application/json'}
    )
    if response.ok:
        print("Action Successful:", response.json())
    else:
        print("Action Failed:", response.status_code, response.text)

# Example Prompt
prompt = "Schedule a Zoom call tomorrow at 2 PM titled 'Team Check-In.'"
execute_mcp_action(prompt)