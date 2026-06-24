import json
import click
from typing import List, Dict, Any, Optional

def find_node(tree: Dict[str, Any], target_name: str) -> Optional[Dict[str, Any]]:
    """Recursively search for a node by name in the nested children tree.
    
    Args:
        tree: The root or current node of the tree.
        target_name: The name to search for.
        
    Returns:
        The matching node or None if not found.
    """
    if str(tree.get("name")).lower() == target_name.lower():
        return tree
    
    children = tree.get("children", [])
    if children:
        for child in children:
            result = find_node(child, target_name)
            if result:
                return result
                
    return None

def collect_all_names(node: Dict[str, Any], names: List[str]) -> None:
    """Recursively collect all 'name' attributes in a node and its descendants.
    
    Args:
        node: The starting node.
        names: The list to append names to.
    """
    if "name" in node:
        names.append(node["name"])
    
    children = node.get("children", [])
    if children:
        for child in children:
            collect_all_names(child, names)

@click.command()
@click.option(
    "--input", 
    type=click.Path(exists=True),
    help="Input UAT JSON file.",
    required=True,
)

@click.option(
    "--root", 
    help="The root concept to extract.",
    required=True,
)

@click.option("--output", 
              help="Output text file.",
              required=True
)

def main(input: str, root: str, output: str) -> None:
    """CLI tool to extract a sub-hierarchy from the UAT JSON ontology."""
    click.echo(f"Loading UAT ontology from {input}...")
    with open(input, "r") as f:
        tree = json.load(f)
        
    click.echo(f"Searching for root node: '{root}'...")
    # The UAT JSON has a top-level 'children' key
    heliophysics_node = find_node(tree, root)
    
    if not heliophysics_node:
        # Check if the root is actually inside the children list directly
        if "children" in tree:
            for child in tree["children"]:
                heliophysics_node = find_node(child, root)
                if heliophysics_node:
                    break
                    
    if not heliophysics_node:
        click.echo(f"Error: Node '{root}' not found in the ontology.")
        return
        
    click.echo(f"Found '{root}'. Collecting all descendants...")
    all_names: List[str] = []
    collect_all_names(heliophysics_node, all_names)
    
    # Sort and unique
    unique_names = sorted(list(set(all_names)))
    
    click.echo(f"Extracted {len(unique_names)} concepts.")
    
    with open(output, "w") as f:
        for name in unique_names:
            f.write(f"{name}\n")
            
    click.echo(f"Results saved to: {output}")

if __name__ == "__main__":
    main()
