<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Switch } from '$lib/components/ui/switch';
	import { KeyValuePairs } from '$lib/components/app';
	import type { KeyValuePair } from '$lib/types';
	import { parseHeadersToArray, serializeHeaders } from '$lib/utils';
	import { UrlProtocol } from '$lib/enums';
	import { MCP_SERVER_URL_PLACEHOLDER } from '$lib/constants';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { CLI_FLAGS } from '$lib/constants';

	import { MCPTransportType } from '$lib/enums';

	interface Props {
		url: string;
		headers: string;
		useProxy?: boolean;
		transport?: MCPTransportType;
		command?: string;
		args?: string[];
		cwd?: string;
		env?: string;
		onUrlChange: (url: string) => void;
		onHeadersChange: (headers: string) => void;
		onUseProxyChange?: (useProxy: boolean) => void;
		onTransportChange?: (transport: MCPTransportType) => void;
		onCommandChange?: (command: string) => void;
		onArgsChange?: (args: string[]) => void;
		onCwdChange?: (cwd: string) => void;
		onEnvChange?: (env: string) => void;
		urlError?: string | null;
		id?: string;
	}

	let {
		url,
		headers,
		useProxy = false,
		transport = MCPTransportType.STREAMABLE_HTTP,
		command = '',
		args = [],
		cwd = '',
		env = '',
		onUrlChange,
		onHeadersChange,
		onUseProxyChange,
		onTransportChange,
		onCommandChange,
		onArgsChange,
		onCwdChange,
		onEnvChange,
		urlError = null,
		id = 'server'
	}: Props = $props();

	let argsString = $state(JSON.stringify(args));
	let envString = $state(env);

	let argsError = $state<string | null>(null);
	let envError = $state<string | null>(null);

	$effect(() => {
		argsString = JSON.stringify(args);
	});

	$effect(() => {
		envString = env;
	});

	function handleArgsBlur() {
		try {
			const parsed = JSON.parse(argsString);
			if (Array.isArray(parsed)) {
				onArgsChange?.(parsed);
				argsError = null;
			} else {
				argsError = 'Arguments must be a JSON array';
			}
		} catch (e) {
			argsError = 'Invalid JSON array';
		}
	}

	function handleEnvBlur() {
		try {
			if (envString.trim()) {
				JSON.parse(envString);
			}
			onEnvChange?.(envString);
			envError = null;
		} catch (e) {
			envError = 'Invalid JSON object';
		}
	}

	let isWebSocket = $derived(
		url.toLowerCase().startsWith(UrlProtocol.WEBSOCKET) ||
			url.toLowerCase().startsWith(UrlProtocol.WEBSOCKET_SECURE)
	);

	let transportValue = $state(transport);
	$effect(() => {
		transportValue = transport;
	});

	function handleTransportChange(v: string) {
		const newTransport = v as MCPTransportType;
		transportValue = newTransport;
		onTransportChange?.(newTransport);
	}

	let headerPairs = $derived<KeyValuePair[]>(parseHeadersToArray(headers));

	function updateHeaderPairs(newPairs: KeyValuePair[]) {
		headerPairs = newPairs;
		onHeadersChange(serializeHeaders(newPairs));
	}
</script>

<div class="grid gap-3">
	{#if mcpStore.isStdioEnabled}
		<div class="mb-2 flex gap-2">
			<Button
				variant={transportValue === MCPTransportType.STREAMABLE_HTTP ? 'default' : 'secondary'}
				size="sm"
				onclick={() => handleTransportChange(MCPTransportType.STREAMABLE_HTTP)}
			>
				Network (HTTP/WS)
			</Button>
			<Button
				variant={transportValue === MCPTransportType.STDIO ? 'default' : 'secondary'}
				size="sm"
				onclick={() => handleTransportChange(MCPTransportType.STDIO)}
			>
				Local (stdio)
			</Button>
		</div>

		{#if transportValue === MCPTransportType.STREAMABLE_HTTP}
			<div class="space-y-4">
				<div>
					<label for="server-url-{id}" class="mb-2 block text-xs font-medium">
						Server URL <span class="text-destructive">*</span>
					</label>

					<Input
						id="server-url-{id}"
						type="url"
						placeholder={MCP_SERVER_URL_PLACEHOLDER}
						value={url}
						oninput={(e) => onUrlChange(e.currentTarget.value)}
						class={urlError ? 'border-destructive' : ''}
					/>

					{#if urlError}
						<p class="mt-1.5 text-xs text-destructive">{urlError}</p>
					{/if}

					{#if !isWebSocket && onUseProxyChange}
						<label
							class={[
								'mt-3 flex items-start gap-2',
								mcpStore.isProxyAvailable && 'cursor-pointer',
								!mcpStore.isProxyAvailable && 'opacity-80'
							]}
						>
							<Switch
								class="mt-1"
								id="use-proxy-{id}"
								checked={useProxy}
								disabled={!mcpStore.isProxyAvailable}
								onCheckedChange={(checked) => onUseProxyChange?.(checked)}
							/>

							<span>
								<span class="text-xs text-muted-foreground">Use llama-server proxy</span>

								<br />

								{#if !mcpStore.isProxyAvailable}
									<span class="inline-flex gap-0.75 text-xs text-muted-foreground/60"
										>(Run <pre>llama-server</pre>
										with
										<pre>{CLI_FLAGS.MCP_PROXY}</pre>
										flag)</span
									>
								{/if}
							</span>
						</label>
					{/if}
				</div>

				<KeyValuePairs
					class="mt-2"
					pairs={headerPairs}
					onPairsChange={updateHeaderPairs}
					keyPlaceholder="Header name"
					valuePlaceholder="Value"
					addButtonLabel="Add"
					emptyMessage="No custom headers configured."
					sectionLabel="Custom Headers"
					sectionLabelOptional
				/>
			</div>
		{:else}
			<div class="space-y-4">
				<div>
					<label for="server-command-{id}" class="mb-2 block text-xs font-medium">
						Command <span class="text-destructive">*</span>
					</label>
					<Input
						id="server-command-{id}"
						placeholder="e.g. node"
						value={command}
						oninput={(e) => onCommandChange?.(e.currentTarget.value)}
					/>
				</div>

				<div>
					<label for="server-args-{id}" class="mb-2 block text-xs font-medium"> Arguments </label>
					<Input
						id="server-args-{id}"
						placeholder='e.g. ["server.js"]'
						bind:value={argsString}
						onblur={handleArgsBlur}
						class={argsError ? 'border-destructive' : ''}
					/>
					{#if argsError}
						<p class="mt-1.5 text-xs text-destructive">{argsError}</p>
					{/if}
				</div>

				<div>
					<label for="server-cwd-{id}" class="mb-2 block text-xs font-medium"> Working Directory </label>
					<Input
						id="server-cwd-{id}"
						placeholder="Optional absolute path"
						value={cwd}
						oninput={(e) => onCwdChange?.(e.currentTarget.value)}
					/>
				</div>

				<div>
					<label for="server-env-{id}" class="mb-2 block text-xs font-medium">
						Environment Variables (JSON)
					</label>
					<Input
						id="server-env-{id}"
						placeholder="e.g. &#123;&quot;KEY&quot;: &quot;VALUE&quot;&#125;"
						bind:value={envString}
						onblur={handleEnvBlur}
						class={envError ? 'border-destructive' : ''}
					/>
					{#if envError}
						<p class="mt-1.5 text-xs text-destructive">{envError}</p>
					{/if}
				</div>
			</div>
		{/if}
	{:else}
		<div>
			<label for="server-url-{id}" class="mb-2 block text-xs font-medium">
				Server URL <span class="text-destructive">*</span>
			</label>

			<Input
				id="server-url-{id}"
				type="url"
				placeholder={MCP_SERVER_URL_PLACEHOLDER}
				value={url}
				oninput={(e) => onUrlChange(e.currentTarget.value)}
				class={urlError ? 'border-destructive' : ''}
			/>

			{#if urlError}
				<p class="mt-1.5 text-xs text-destructive">{urlError}</p>
			{/if}

			{#if !isWebSocket && onUseProxyChange}
				<label
					class={[
						'mt-3 flex items-start gap-2',
						mcpStore.isProxyAvailable && 'cursor-pointer',
						!mcpStore.isProxyAvailable && 'opacity-80'
					]}
				>
					<Switch
						class="mt-1"
						id="use-proxy-{id}"
						checked={useProxy}
						disabled={!mcpStore.isProxyAvailable}
						onCheckedChange={(checked) => onUseProxyChange?.(checked)}
					/>

					<span>
						<span class="text-xs text-muted-foreground">Use llama-server proxy</span>

						<br />

						{#if !mcpStore.isProxyAvailable}
							<span class="inline-flex gap-0.75 text-xs text-muted-foreground/60"
								>(Run <pre>llama-server</pre>
								with
								<pre>{CLI_FLAGS.MCP_PROXY}</pre>
								flag). (If stdio is desired, ensure <pre>--ui-mcp-stdio</pre> is not disabled)</span
							>
						{/if}
				</span>
				</label>
			{/if}
		</div>

		<KeyValuePairs
			class="mt-2"
			pairs={headerPairs}
			onPairsChange={updateHeaderPairs}
			keyPlaceholder="Header name"
			valuePlaceholder="Value"
			addButtonLabel="Add"
			emptyMessage="No custom headers configured."
			sectionLabel="Custom Headers"
			sectionLabelOptional
		/>
	{/if}
</div>
