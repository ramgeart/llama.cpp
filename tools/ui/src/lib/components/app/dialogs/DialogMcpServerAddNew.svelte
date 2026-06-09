<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { McpServerForm } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { uuid } from '$lib/utils';
	import { MCP_SERVER_ID_PREFIX } from '$lib/constants';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	import { MCPTransportType } from '$lib/enums';

	let newServerUrl = $state('');
	let newServerHeaders = $state('');
	let newServerTransport = $state(MCPTransportType.STREAMABLE_HTTP);
	let newServerCommand = $state('');
	let newServerArgs = $state<string[]>([]);
	let newServerCwd = $state('');
	let newServerEnv = $state('');

	let newServerUrlError = $derived.by(() => {
		if (newServerTransport === MCPTransportType.STDIO) return null;
		if (!newServerUrl.trim()) return 'URL is required';
		try {
			new URL(newServerUrl);

			return null;
		} catch {
			return 'Invalid URL format';
		}
	});

	let canSave = $derived(
		!newServerUrlError &&
			(newServerTransport !== MCPTransportType.STDIO || newServerCommand.trim())
	);

	function handleOpenChange(value: boolean) {
		if (!value) {
			newServerUrl = '';
			newServerHeaders = '';
			newServerTransport = MCPTransportType.STREAMABLE_HTTP;
			newServerCommand = '';
			newServerArgs = [];
			newServerCwd = '';
			newServerEnv = '';
		}
		open = value;
		onOpenChange?.(value);
	}

	function saveNewServer() {
		if (!canSave) return;

		const newServerId = uuid() ?? `${MCP_SERVER_ID_PREFIX}-${Date.now()}`;

		mcpStore.addServer({
			id: newServerId,
			enabled: true,
			url: newServerTransport === MCPTransportType.STDIO ? undefined : newServerUrl.trim(),
			headers: newServerHeaders.trim() || undefined,
			transport: newServerTransport,
			command: newServerCommand.trim() || undefined,
			args: newServerArgs.length > 0 ? newServerArgs : undefined,
			cwd: newServerCwd.trim() || undefined,
			env: newServerEnv.trim() || undefined
		});

		conversationsStore.setMcpServerOverride(newServerId, true);

		handleOpenChange(false);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-md">
		<Dialog.Header>
			<Dialog.Title>Add New Server</Dialog.Title>
		</Dialog.Header>

		<div class="space-y-4 py-4">
			<McpServerForm
				url={newServerUrl}
				headers={newServerHeaders}
				transport={newServerTransport}
				command={newServerCommand}
				args={newServerArgs}
				cwd={newServerCwd}
				env={newServerEnv}
				onUrlChange={(v) => (newServerUrl = v)}
				onHeadersChange={(v) => (newServerHeaders = v)}
				onTransportChange={(v) => (newServerTransport = v)}
				onCommandChange={(v) => (newServerCommand = v)}
				onArgsChange={(v) => (newServerArgs = v)}
				onCwdChange={(v) => (newServerCwd = v)}
				onEnvChange={(v) => (newServerEnv = v)}
				urlError={newServerUrl ? newServerUrlError : null}
				id="new-server"
			/>
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button
				variant="default"
				size="sm"
				onclick={saveNewServer}
				disabled={!canSave}
				aria-label="Save"
			>
				Add
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
