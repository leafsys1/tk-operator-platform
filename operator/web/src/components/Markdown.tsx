import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** agent 答复的 Markdown 渲染（GFM：表格 / 任务列表 / 删除线）。
 *
 * 不开 raw HTML（react-markdown 默认就不渲染 HTML），避免 agent 输出里夹带 HTML 注入；
 * 链接统一新窗口打开。样式在 `index.css` 的 `.md` 段。
 */
export default function Markdown({ text, className = "" }: { text: string; className?: string }) {
  return (
    <div className={`md ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, title, children }) => (
            <a href={href} title={title} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
