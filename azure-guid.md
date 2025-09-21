# Azure Cloud Developer Reference Guide (2025)

**A comprehensive, up-to-date reference for developers working daily with Microsoft Azure. Includes advanced, intermediate, IaC, security, networking, serverless, AI, data engineering, databases, Python/TypeScript code, and Terraform.**

---

## Table of Contents

- [Overview](#overview)
- [Key Azure Services (2025)](#key-azure-services)
- [Infrastructure as Code (IaC) with Terraform](#infrastructure-as-code-iac-with-terraform)
- [Security Best Practices](#security-best-practices)
- [Azure Networking](#azure-networking)
- [Serverless Computing](#serverless-computing)
- [AI & Machine Learning](#ai--machine-learning)
- [Data Engineering](#data-engineering)
- [Database Services](#database-services)
- [Python and TypeScript Azure SDK](#python-and-typescript-sdks)
- [Reference Links & Sources](#reference-links--sources)

---

## Overview

Microsoft Azure in 2025 delivers extensive, cloud-native, developer-centric features for building, deploying, and scaling modern applications. This doc showcases service highlights, code samples in Python and TypeScript, Terraform templates, and the latest practices in security, networking, serverless, AI, and data engineering.

---

## Key Azure Services (2025)

- **App Services, Functions, Logic Apps, Container Apps** (for compute, event, API, or workflow automation)
- **Azure Arc, Azure Stack, Azure DevOps, GitHub Actions** (multi-cloud, hybrid, and DevOps workflows)
- **AI & Data:** Azure AI Studio, Azure OpenAI, Azure Databricks, Synapse Analytics, Azure Fabric, Azure Data Lake Gen 3, Purview, Event Hubs, Stream Analytics
- **Networking:** VNet, Private Link/Endpoint, ExpressRoute, Virtual WAN, Load Balancer, Firewall, Application Gateway, DDoS Protection
- **Security:** Defender for Cloud, Sentinel, Azure AD, RBAC, PIM, Key Vault
- **Databases:** Azure SQL, Cosmos DB, PostgreSQL, MySQL, Redis, Managed Instances

---

## Infrastructure as Code (IaC) with Terraform

Provision and manage Azure infrastructure using [Terraform](https://learn.microsoft.com/en-us/azure/developer/terraform/).

```hcl
# Example: Provision Azure Resource Group, Virtual Network, and Storage
provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "example" {
  name     = "example-resources"
  location = "East US"
}

resource "azurerm_virtual_network" "example" {
  name                = "example-network"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name
}

resource "azurerm_storage_account" "example" {
  name                     = "examplestoracc"
  resource_group_name      = azurerm_resource_group.example.name
  location                 = azurerm_resource_group.example.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

# Run: terraform init, then terraform apply
```
---

## Security Best Practices

- **Identity:** Enforce MFA, minimal roles with RBAC, use PIM for just-in-time access
- **Threat Protection:** Enable Defender for Cloud, Azure Sentinel for SIEM/SOAR, continuous vulnerability scanning
- **Network:** Apply NSGs, use Private Link, ExpressRoute and Virtual WAN for secure/private connectivity
- **Zero Trust:** Segment networks, encrypt data in transit and at rest (Key Vault for secrets)
- **DevOps:** Secure supply chain, integrate security in CI/CD, enable GitHub/Azure DevOps security scanning

*Reference: [Azure Security](https://learn.microsoft.com/en-us/azure/security/fundamentals/best-practices-and-patterns)*

---

## Azure Networking

- **VNet/Peering:** Build isolated, secure networks for resources
- **ExpressRoute:** Dedicated circuits between Azure and on-prem
- **Private Link:** Connect privately to Azure PaaS (SQL, Storage, etc.)
- **Virtual WAN:** Centralized networking across regions/offices
- **Firewall/Application Gateway:** Layer 7 security and WAF
- **DDoS/Protection:** Built-in mitigation for internet-facing endpoints

```python
# Python: Create/Manage Azure VNet
from azure.mgmt.network import NetworkManagementClient
from azure.identity import DefaultAzureCredential

network_client = NetworkManagementClient(DefaultAzureCredential(), "<subscription_id>")

result = network_client.virtual_networks.begin_create_or_update(
    "RESOURCE_GROUP",
    "vnet-dev",
    {
        "location": "eastus",
        "address_space": {"address_prefixes": ["10.0.0.0/16"]},
    },
)
```

---

## Serverless Computing

Azure offers Function as a Service (FaaS), Logic Apps, Event Grid, serverless containers, and native event-driven patterns.

- **Azure Functions:** Autoscaling event handlers (HTTP, queue, timer)
    - Hot/cold plan, Linux/Windows, managed identity

**Python Example – Hello Function:**
```python
import logging
def main(req):
    logging.info(f"Request received: {req}")
    return "Hello from Azure Function!"
```

**TypeScript Example – HTTP Function:**
```typescript
import { AzureFunction, Context, HttpRequest } from "@azure/functions"

const httpTrigger: AzureFunction = async function (context: Context, req: HttpRequest): Promise<void> {
    context.res = {
        body: "Hello from Azure Functions!"
    };
};
export default httpTrigger;
```

**Terraform Example – Azure Function App:**
```hcl
resource "azurerm_function_app" "example" {
  name                       = "example-func-app"
  location                   = azurerm_resource_group.example.location
  resource_group_name        = azurerm_resource_group.example.name
  app_service_plan_id        = azurerm_app_service_plan.example.id
  storage_account_name       = azurerm_storage_account.example.name
  storage_account_access_key = azurerm_storage_account.example.primary_access_key
  version = "~4"
}
```
---

## AI & Machine Learning

Azure offers integrated AI/ML tools for developers:
- **Azure AI Studio, OpenAI on Azure, Cognitive Services (Vision, Text, Speech), Azure ML, Synapse ML, Databricks, Data Explorer.**

**Python Example – Deploying a ML model:**
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
ml_client = MLClient(DefaultAzureCredential(), '<subscription_id>', '<resource_group>', '<workspace_name>')
registered_model = ml_client.models.create_or_update(
    name="my-model",
    path="local-model-dir/",
    description="A sample ML model on Azure"
)
```

**TypeScript Example – Azure AI Text Analytics:**
```typescript
import { TextAnalyticsClient, AzureKeyCredential } from "@azure/ai-text-analytics";
const client = new TextAnalyticsClient(endpoint, new AzureKeyCredential(apiKey));
const result = await client.analyzeSentiment(["Azure is awesome!"]);
```

---

## Data Engineering

- **Modern Recommended Tools (2025):**
    - Microsoft Fabric, Synapse Analytics, Azure Data Lake Gen 3, Databricks, Data Factory (serverless mapping flows, pipelines), Event Hubs, Stream Analytics, Purview, Cosmos DB
- **Key patterns:**
    - Adopt lakehouse architecture, 
    - serverless/event-driven ETL (Event Grid, Functions, Data Factory pipelines)
    - AI-augmented DataOps (Purview, Copilot, ML-powered quality)
    - Real-time analytics (Stream Analytics, Data Explorer)
    - Secure, automate, CI/CD pipelines, version data assets with Delta Lake

---

## Database Services

- **Azure SQL (PaaS, Managed Instance, and VM), Cosmos DB, PostgreSQL, MySQL, MariaDB, Redis, Table Storage**
- **Features:** Global availability, high perf, HA, geo-replication, backup, built-in security

```python
# Python Example: Query from SQL Database
import pyodbc
conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=<server>;DATABASE=<db>;UID=<user>;PWD=<password>')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sys.tables")
for row in cursor.fetchall():
    print(row)
```

```typescript
// TypeScript Example: Query Cosmos DB
import { CosmosClient } from "@azure/cosmos"
const client = new CosmosClient({ endpoint, key });
const { database } = await client.databases.createIfNotExists({ id: "mydb" });
```

---

## Python and TypeScript SDKs

- [Azure SDK for Python](https://learn.microsoft.com/en-us/azure/developer/python/)
- [Azure SDK for TypeScript/JavaScript](https://learn.microsoft.com/en-us/azure/developer/javascript/)
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/)
- [Client references, samples, and guides on GitHub](https://github.com/Azure/azure-sdk-for-python), [TypeScript samples](https://github.com/Azure/azure-sdk-for-js-samples)

---

## Reference Links & Sources

- [Microsoft Azure Docs](https://learn.microsoft.com/en-us/azure/)
- [What's New – Azure Dev Docs](https://learn.microsoft.com/en-us/azure/developer/intro/whats-new)
- [Terraform on Azure](https://learn.microsoft.com/en-us/azure/developer/terraform/)
- [Azure Security Best Practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/best-practices-and-patterns)
- [Azure Networking](https://learn.microsoft.com/en-us/azure/networking/fundamentals/networking-overview)
- [Azure Databases Overview](https://learn.microsoft.com/en-us/azure/azure-sql/database/sql-database-paas-overview?view=azuresql)
- [Azure Python SDK](https://learn.microsoft.com/en-us/azure/developer/python/sdk/azure-sdk-overview)
- [Azure TypeScript SDK](https://learn.microsoft.com/en-us/azure/developer/javascript/)
- [GitHub – Azure Terraform Examples](https://github.com/alfonsof/terraform-azure-examples)
- [Serverless Azure Docs](https://learn.microsoft.com/en-us/azure/azure-functions/)
- [AI and ML Services](https://azure.microsoft.com/en-us/products/category/ai)
- [Data Engineering on Azure](https://learn.microsoft.com/en-us/training/paths/get-started-data-engineering/)
- [Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/)

---

_Last updated: 2025-09-21_

---

> Contributions welcome! Edit this file in your repo and submit pull requests as Azure/new capabilities shift.

