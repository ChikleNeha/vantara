// jsdom lacks ResizeObserver, which recharts' ResponsiveContainer needs.
class RO { observe() {} unobserve() {} disconnect() {} }
globalThis.ResizeObserver = globalThis.ResizeObserver || RO
