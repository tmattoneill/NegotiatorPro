"""
Docker infrastructure tests for NegotiatorPro
"""
import os
import pytest
import subprocess
import yaml
from pathlib import Path


class TestDockerfile:
    """Test Dockerfile configuration"""

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists"""
        assert Path("Dockerfile").exists(), "Dockerfile not found"

    def test_dockerfile_syntax(self):
        """Verify Dockerfile has valid syntax"""
        dockerfile_path = Path("Dockerfile")
        content = dockerfile_path.read_text()

        # Check for required directives
        assert "FROM" in content, "Dockerfile missing FROM directive"
        assert "WORKDIR" in content, "Dockerfile missing WORKDIR directive"
        assert "COPY" in content or "ADD" in content, "Dockerfile missing COPY/ADD directive"
        assert "CMD" in content or "ENTRYPOINT" in content, "Dockerfile missing CMD/ENTRYPOINT"

    def test_dockerfile_uses_multistage_build(self):
        """Verify Dockerfile uses multi-stage build for optimization"""
        content = Path("Dockerfile").read_text()
        from_count = content.count("FROM ")
        assert from_count >= 2, "Dockerfile should use multi-stage build"

    def test_dockerfile_uses_non_root_user(self):
        """Verify Dockerfile creates and uses non-root user"""
        content = Path("Dockerfile").read_text()
        assert "USER " in content, "Dockerfile should specify non-root user"
        assert "useradd" in content or "RUN adduser" in content, "Dockerfile should create user"

    def test_dockerfile_has_healthcheck(self):
        """Verify Dockerfile includes healthcheck"""
        content = Path("Dockerfile").read_text()
        assert "HEALTHCHECK" in content, "Dockerfile should include HEALTHCHECK"

    def test_dockerfile_exposes_port(self):
        """Verify Dockerfile exposes correct port"""
        content = Path("Dockerfile").read_text()
        assert "EXPOSE 7860" in content, "Dockerfile should expose port 7860"

    def test_dockerfile_sets_env_variables(self):
        """Verify Dockerfile sets required environment variables"""
        content = Path("Dockerfile").read_text()
        assert "ENV" in content, "Dockerfile should set environment variables"


class TestDockerCompose:
    """Test docker-compose.yml configuration"""

    def test_docker_compose_exists(self):
        """Verify docker-compose.yml exists"""
        assert Path("docker-compose.yml").exists(), "docker-compose.yml not found"

    def test_docker_compose_syntax(self):
        """Verify docker-compose.yml has valid YAML syntax"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        assert config is not None, "docker-compose.yml is empty or invalid"
        assert "services" in config, "docker-compose.yml missing services section"

    def test_docker_compose_version(self):
        """Verify docker-compose.yml has version specified"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        assert "version" in config, "docker-compose.yml should specify version"

    def test_docker_compose_service_defined(self):
        """Verify negotiator-pro service is defined"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        assert "negotiator-pro" in config["services"], "negotiator-pro service not defined"

    def test_docker_compose_port_mapping(self):
        """Verify port mapping is configured"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        service = config["services"]["negotiator-pro"]
        assert "ports" in service, "Service missing port mapping"
        assert any("7860" in str(port) for port in service["ports"]), "Port 7860 not mapped"

    def test_docker_compose_volumes(self):
        """Verify persistent volumes are configured"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        service = config["services"]["negotiator-pro"]
        assert "volumes" in service, "Service missing volume configuration"

        volumes = service["volumes"]
        required_volumes = ["vectorstore", "uploads", "sources"]
        for vol in required_volumes:
            assert any(vol in str(v) for v in volumes), f"Missing {vol} volume"

    def test_docker_compose_environment_variables(self):
        """Verify environment variables are configured"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        service = config["services"]["negotiator-pro"]
        assert "environment" in service, "Service missing environment variables"

        env = service["environment"]
        assert any("OPENAI_API_KEY" in str(e) for e in env), "Missing OPENAI_API_KEY"

    def test_docker_compose_restart_policy(self):
        """Verify restart policy is configured"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        service = config["services"]["negotiator-pro"]
        assert "restart" in service, "Service missing restart policy"

    def test_docker_compose_healthcheck(self):
        """Verify healthcheck is configured"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        service = config["services"]["negotiator-pro"]
        assert "healthcheck" in service, "Service missing healthcheck"

    def test_docker_compose_resource_limits(self):
        """Verify resource limits are configured"""
        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        service = config["services"]["negotiator-pro"]
        if "deploy" in service:
            assert "resources" in service["deploy"], "Service missing resource limits"


class TestDockerIgnore:
    """Test .dockerignore configuration"""

    def test_dockerignore_exists(self):
        """Verify .dockerignore exists"""
        assert Path(".dockerignore").exists(), ".dockerignore not found"

    def test_dockerignore_excludes_venv(self):
        """Verify .dockerignore excludes virtual environment"""
        content = Path(".dockerignore").read_text()
        assert ".venv" in content or "venv/" in content, ".dockerignore should exclude venv"

    def test_dockerignore_excludes_git(self):
        """Verify .dockerignore excludes git files"""
        content = Path(".dockerignore").read_text()
        assert ".git" in content, ".dockerignore should exclude .git"

    def test_dockerignore_excludes_pycache(self):
        """Verify .dockerignore excludes __pycache__"""
        content = Path(".dockerignore").read_text()
        assert "__pycache__" in content, ".dockerignore should exclude __pycache__"

    def test_dockerignore_excludes_env_files(self):
        """Verify .dockerignore excludes .env files"""
        content = Path(".dockerignore").read_text()
        assert ".env" in content, ".dockerignore should exclude .env files"


class TestDockerBuild:
    """Test Docker build process"""

    @pytest.mark.skipif(
        not os.path.exists("/var/run/docker.sock"),
        reason="Docker daemon not available"
    )
    def test_docker_build_succeeds(self):
        """Test that Docker image builds successfully"""
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "negotiator-pro-test", "."],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        except subprocess.TimeoutExpired:
            pytest.fail("Docker build timed out after 5 minutes")
        except FileNotFoundError:
            pytest.skip("Docker command not found")

    @pytest.mark.skipif(
        not os.path.exists("/var/run/docker.sock"),
        reason="Docker daemon not available"
    )
    def test_docker_compose_config_valid(self):
        """Test that docker-compose config is valid"""
        try:
            result = subprocess.run(
                ["docker", "compose", "config"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"docker-compose config invalid: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("docker-compose command not found")


class TestDeploymentDocs:
    """Test deployment documentation"""

    def test_deployment_md_exists(self):
        """Verify DEPLOYMENT.md exists"""
        assert Path("DEPLOYMENT.md").exists(), "DEPLOYMENT.md not found"

    def test_docker_deploy_md_exists(self):
        """Verify DOCKER-DEPLOY.md exists"""
        assert Path("DOCKER-DEPLOY.md").exists(), "DOCKER-DEPLOY.md not found"

    def test_deployment_docs_have_docker_instructions(self):
        """Verify deployment docs include Docker instructions"""
        deployment_content = Path("DEPLOYMENT.md").read_text()
        assert "docker" in deployment_content.lower(), "DEPLOYMENT.md missing Docker instructions"
        assert "docker compose" in deployment_content.lower(), "DEPLOYMENT.md missing docker-compose instructions"

    def test_docker_deploy_has_quick_start(self):
        """Verify DOCKER-DEPLOY.md has quick start guide"""
        content = Path("DOCKER-DEPLOY.md").read_text()
        assert "docker compose up" in content.lower(), "DOCKER-DEPLOY.md missing docker compose up"
        assert ".env" in content.lower(), "DOCKER-DEPLOY.md missing .env setup"
