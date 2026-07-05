"""Gemini CLI agent variant that reuses a preinstalled CLI when available."""

from __future__ import annotations

import os
import shlex

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


DEFAULT_GEMINI_RUN_TIMEOUT_SEC = "4500"


class CachedGeminiCli(GeminiCli):
    """Skip Gemini CLI network install when the environment already has it.

    Harbor's built-in Gemini agent installs nvm, Node, and @google/gemini-cli
    inside every trial container. That makes each material vulnerable to npm,
    nodejs.org, and GitHub flakes. This subclass keeps the same run behavior but
    first checks whether `gemini` is already available through either PATH or
    ~/.nvm/nvm.sh.
    """

    async def install(self, environment: BaseEnvironment) -> None:
        probe = await self.exec_as_agent(
            environment,
            command=(
                "if command -v gemini >/dev/null 2>&1; then "
                "  echo HARBOR_GEMINI_CLI_READY; gemini --version; "
                "elif [ -s ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh && "
                "     command -v gemini >/dev/null 2>&1; then "
                "  echo HARBOR_GEMINI_CLI_READY; gemini --version; "
                "else "
                "  echo HARBOR_GEMINI_CLI_MISSING; "
                "fi"
            ),
        )
        if "HARBOR_GEMINI_CLI_READY" in (probe.stdout or ""):
            await self.exec_as_agent(
                environment,
                command=(
                    "mkdir -p ~/.nvm && "
                    "printf '%s\n' '# Harbor compatibility shim for preinstalled Gemini CLI' "
                    "'export PATH=/usr/local/bin:/usr/bin:/bin:$PATH' "
                    "> ~/.nvm/nvm.sh"
                ),
            )
            await self.exec_as_agent(
                environment,
                command=(
                    "mkdir -p ~/.gemini && "
                    "cat > ~/.gemini/settings.json << 'SETTINGS'\n"
                    '{\n  "experimental": {\n    "skills": true\n  }\n}\n'
                    "SETTINGS"
                ),
            )
            return

        await super().install(environment)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run Gemini CLI with an inner timeout so fetch stalls fail cleanly."""
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        model = self.model_name.split("/")[-1]
        env = {"GEMINI_CLI_TRUST_WORKSPACE": "true"}

        oauth_creds_path = self._resolve_oauth_creds_path()
        use_oauth = oauth_creds_path is not None
        auth_type = "oauth-personal" if use_oauth else self._resolve_env_auth_type()

        if use_oauth:
            for var in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
                value = self._get_env(var)
                if value:
                    env[var] = value
        else:
            self.logger.debug("Gemini auth: using API key / env credentials")
            auth_vars = [
                "GEMINI_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_API_KEY",
            ]
            for var in auth_vars:
                if var in os.environ:
                    env[var] = os.environ[var]

        run_timeout_sec = os.environ.get(
            "HARBOR_GEMINI_RUN_TIMEOUT_SEC", DEFAULT_GEMINI_RUN_TIMEOUT_SEC
        )
        env["HARBOR_GEMINI_RUN_TIMEOUT_SEC"] = run_timeout_sec

        if use_oauth and oauth_creds_path is not None:
            await self._inject_oauth_creds(environment, oauth_creds_path, env)

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env)

        settings_command, model_alias = self._build_settings_command(
            model, auth_type=auth_type
        )
        if settings_command:
            await self.exec_as_agent(environment, command=settings_command, env=env)

        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""
        run_model = shlex.quote(model_alias or model)

        wrapper = os.environ.get("HARBOR_AGENT_COMMAND_WRAPPER", "").strip()
        wrapper_prefix = ""
        if wrapper:
            wrapper_prefix = f"{shlex.quote(wrapper)} "
            env["HARBOR_AGENT_COMMAND_WRAPPER"] = wrapper

        self.logger.info("HARBOR_AGENT_COMMAND_WRAPPER=%r", wrapper)
        
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "set -o pipefail; "
                    ". ~/.nvm/nvm.sh; "
                    "timeout --kill-after=10s "
                    '"${HARBOR_GEMINI_RUN_TIMEOUT_SEC:-600}s" '
                    f"{wrapper_prefix}gemini --yolo {extra_flags}--model={run_model} "
                    f"--prompt={escaped_instruction} "
                    "2>&1 </dev/null | stdbuf -oL tee /logs/agent/gemini-cli.txt"
                ),
                env=env,
            )
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        "src=$(find ~/.gemini/tmp -type f "
                        "\\( -name 'session-*.jsonl' -o -name 'session-*.json' \\) "
                        "-printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -n1 "
                        "| awk '{print $2}'); "
                        'if [ -n "$src" ]; then '
                        'cp "$src" "/logs/agent/gemini-cli.trajectory.${src##*.}"; '
                        "fi"
                    ),
                )
            except Exception:
                pass
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"rm -rf {shlex.quote(self._REMOTE_SECRETS_DIR.as_posix())} "
                        "~/.gemini/oauth_creds.json"
                    ),
                    env=env,
                )
            except Exception:
                pass
