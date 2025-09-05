# Elastic Beanstalk Deployment (Docker Platform)

## Variante A – EB baut das Docker-Image aus dem Repo (einfach)

### Voraussetzungen (lokal)
- AWS CLI & EB CLI: `pip install awscli awsebcli --upgrade`
- AWS Credentials konfiguriert: `aws configure`

### Einmalig
1. **Init**  
   `eb init <APPLICATION_NAME> --platform "Docker" --region <AWS_REGION>`
2. **Umgebung**  
   `eb create <ENVIRONMENT_NAME> --single`

### Deployment
- Manuell: `eb deploy`
- Automatisch: GitHub Actions Workflow `ci-cd.yml` (Job *deploy-eb*) – setzt Secrets voraus.

---

## Variante B – EB zieht ein vorgebautes Image (Dockerrun.aws.json)

1. Baue & pushe das Image in GHCR/ECR (CI-Job `build-and-push` erledigt GHCR).
2. Erstelle `Dockerrun.aws.json` im Repo-Root:

```json
{
  "AWSEBDockerrunVersion": 1,
  "Image": { "Name": "ghcr.io/SimonGithub00/heart-ml:latest", "Update": "true" },
  "Ports": [{ "ContainerPort": "8080" }],
  "Logging": "/var/log/app"
}
