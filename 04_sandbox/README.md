# Setting up sandbox example

## On the host machine

`mkdir ~/ai-work`

`chmod 700 ~/ai-work`

`cd ai-work`

`mkdir projects`

`curl -fsSL https://install.microsandbox.dev | sh`

### Project example

`mkdir projects/project1`

`echo "hello from MacOS" > projects/project1/host-file.txt`

`msb create ubuntu:24.04 --name pi-agent --workdir /workspace  --net "public,host" -v "$HOME/ai-work/projects/project1:/workspace"`

`msb exec pi-agent -- /bin/bash`

## In the sandbox VM

`apt-get update`

`apt-get install -y git curl ripgrep ca-certificates build-essential sudo`

```ssh
cat > /usr/local/bin/run-pi <<'EOF'
#!/bin/bash

export NVM_DIR="/home/agent/.nvm"

if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    echo "NVM not found at $NVM_DIR" >&2
    exit 1
fi

source "$NVM_DIR/nvm.sh"

cd /workspace || exit 1

exec pi "$@"
EOF

chmod 755 /usr/local/bin/run-pi
```

`useradd -m -s /bin/bash agent`

`chown -R agent:agent /workspace`

`su - agent`

`curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash`

```ssh
export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
  [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
```

`nvm install 22`

`npm install -g --ignore-scripts @earendil-works/pi-coding-agent`

`pi --version`

`pi`


## exiting

To exit pi `^+D`

To exit VM user `exit`

## running pi

`msb exec pi-agent --user agent -- run-pi`

or

`msb exec pi-agent -- su - agent -c 'run-pi'`

# Setting up local model

## On host machine.

`curl -fsSL https://ollama.com/install.sh | sh`

`ollama pull qwen3-coder:30b`

Testing without pi

`ollama run qwen3-coder:30b`

To exit

`/bye`

## In the sandbox VM

[Configure Ollama](https://pi.dev/docs/latest/models#minimal-example) as a custom provider in `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://host.microsandbox.internal:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        { "id": "qwen3-coder:30b" }
      ]
    }
  }
}
```

`pi`

# Explore further

[Pi](https://pi.dev/docs/latest)

[microsandbox](https://docs.microsandbox.dev/getting-started/introduction)

[Ollama](https://docs.ollama.com/api/introduction)

[Qwen3-coder](https://qwen.ai/blog?id=qwen3-coder)
