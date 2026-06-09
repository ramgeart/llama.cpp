<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { McpServerForm } from '$lib/components/app/mcp';

	import { MCPTransportType } from '$lib/enums';
	import type { MCPServerSettingsEntry } from '$lib/types';

	interface Props {
		server: MCPServerSettingsEntry;
		onSave: (updates: Partial<MCPServerSettingsEntry>) => void;
		onCancel: () => void;
	}

	let { server, onSave, onCancel }: Props = $props();

	let editUrl = $state('');
	let editHeaders = $state('');
	let editUseProxy = $state(false);
	let editTransport = $state(MCPTransportType.STREAMABLE_HTTP);
	let editCommand = $state('');
	let editArgs = $state<string[]>([]);
	let editCwd = $state('');
	let editEnv = $state('');

	$effect(() => {
		editUrl = server.url ?? '';
		editHeaders = server.headers ?? '';
		editUseProxy = server.useProxy ?? false;
		editTransport = server.transport ?? MCPTransportType.STREAMABLE_HTTP;
		editCommand = server.command ?? '';
		editArgs = server.args ?? [];
		editCwd = server.cwd ?? '';
		editEnv = server.env ?? '';
	});

	let urlError = $derived.by(() => {
		if (editTransport === MCPTransportType.STDIO) return null;
		if (!editUrl.trim()) return 'URL is required';
		try {
			new URL(editUrl);
			return null;
		} catch {
			return 'Invalid URL format';
		}
	});

	let canSave = $derived(!urlError && (editTransport !== MCPTransportType.STDIO || editCommand.trim()));

	function handleSave() {
		if (!canSave) return;
		onSave({
			url: editUrl.trim(),
			headers: editHeaders.trim(),
			useProxy: editUseProxy,
			transport: editTransport,
			command: editCommand.trim(),
			args: editArgs,
			cwd: editCwd.trim(),
			env: editEnv.trim()
		});
	}
</script>

<div class="space-y-4">
	<p class="font-medium">Configure Server</p>

	<McpServerForm
		url={editUrl}
		headers={editHeaders}
		useProxy={editUseProxy}
			transport={editTransport}
			command={editCommand}
			args={editArgs}
			cwd={editCwd}
			env={editEnv}
		onUrlChange={(v) => (editUrl = v)}
		onHeadersChange={(v) => (editHeaders = v)}
		onUseProxyChange={(v) => (editUseProxy = v)}
			onTransportChange={(v) => (editTransport = v)}
			onCommandChange={(v) => (editCommand = v)}
			onArgsChange={(v) => (editArgs = v)}
			onCwdChange={(v) => (editCwd = v)}
			onEnvChange={(v) => (editEnv = v)}
		urlError={editUrl ? urlError : null}
		id={server.id}
	/>

	<div class="flex items-center justify-end gap-2">
		<Button variant="secondary" size="sm" onclick={onCancel}>Cancel</Button>

		<Button size="sm" onclick={handleSave} disabled={!canSave}>
			{server.url?.trim() || server.command?.trim() ? 'Update' : 'Add'}
		</Button>
	</div>
</div>
